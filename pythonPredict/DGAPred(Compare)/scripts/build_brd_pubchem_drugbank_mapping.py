"""构建当前训练药物 BRD 到 PubChem / DrugBank 的严格映射表。

执行顺序严格按当前需求：
1. 先从多种精确渠道获取 PubChem CID
2. 再用原有本地方式获取 DrugBank ID
3. 最后用 DrugBank-PubChem 桥接表做一致性校验和补充

匹配原则：
- 只接受完全精确匹配
- 任一渠道返回多个不同候选时，视为歧义，不自动写入
- 不做模糊匹配，不做近似匹配
"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


PUBCHEM_TIMEOUT = 30
PUBCHEM_SLEEP_SECONDS = 0.05
PUBCHEM_CACHE = {}


def normalize_text(value):
    """统一基础文本格式。"""
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_upper(value):
    """统一大写格式，适合 InChIKey。"""
    return normalize_text(value).upper()


def normalize_lower(value):
    """统一小写格式，适合名字精确匹配。"""
    return normalize_text(value).lower()


def split_exact_aliases(value):
    """把别名拆成精确词条，不做任何模糊清洗。"""
    text = normalize_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def load_current_drugs(project_feature_dir):
    """读取当前训练样本里真正参与训练的 BRD 顺序。"""
    drug_side = pd.read_csv(project_feature_dir / "drug_side.csv")
    return drug_side["pert_id"].map(normalize_text).tolist()


def load_base_info(current_drugs, compoundinfo_path):
    """读取 BRD 的基础结构信息。"""
    compound = pd.read_csv(compoundinfo_path, sep="\t", low_memory=False)
    compound = compound[compound["pert_id"].isin(current_drugs)].copy()

    compound["pert_id"] = compound["pert_id"].map(normalize_text)
    compound["cmap_name"] = compound["cmap_name"].map(normalize_text)
    compound["canonical_smiles"] = compound["canonical_smiles"].map(normalize_text)
    compound["inchi_key"] = compound["inchi_key"].map(normalize_upper)
    compound["compound_aliases"] = compound["compound_aliases"].map(normalize_text)

    # 一条 BRD 可能出现多次，优先保留信息最完整的那行。
    compound["info_score"] = (
        (compound["canonical_smiles"] != "").astype(int)
        + (compound["inchi_key"] != "").astype(int)
        + (compound["cmap_name"] != "").astype(int)
        + (compound["compound_aliases"] != "").astype(int)
    )
    compound = compound.sort_values(["pert_id", "info_score"], ascending=[True, False])
    compound = compound.drop_duplicates(subset=["pert_id"], keep="first")

    result = {}
    for _, row in compound.iterrows():
        result[row["pert_id"]] = {
            "cmap_name": row["cmap_name"],
            "canonical_smiles": row["canonical_smiles"],
            "inchi_key": row["inchi_key"],
            "compound_aliases": row["compound_aliases"],
        }
    return result


def build_local_pubchem_name_index(pubchem_drugbank_path):
    """从现有 DrugBank-PubChem 桥接表构建本地 PubChem 名称索引。"""
    df = pd.read_csv(pubchem_drugbank_path, low_memory=False)
    df["drugbank_id"] = df["drugbank_id"].map(normalize_text)
    df["name"] = df["name"].map(normalize_text)
    df["pubchem_Compound_CID"] = df["pubchem_Compound_CID"].map(normalize_text)
    df["pubchem_drug_name"] = df["pubchem_drug_name"].map(normalize_text)

    name_to_cids = {}
    for _, row in df.iterrows():
        cid = row["pubchem_Compound_CID"]
        if not cid or cid == "-":
            continue
        for name in [row["name"], row["pubchem_drug_name"]]:
            key = normalize_lower(name)
            if key:
                name_to_cids.setdefault(key, set()).add(cid)
    return name_to_cids


def build_drugbank_indexes(drugbank_vocab_path, pubchem_drugbank_path):
    """构建本地 DrugBank 精确匹配索引。"""
    vocab = pd.read_csv(drugbank_vocab_path, low_memory=False)
    vocab["DrugBank ID"] = vocab["DrugBank ID"].map(normalize_text)
    vocab["Common name"] = vocab["Common name"].map(normalize_text)
    vocab["Standard InChI Key"] = vocab["Standard InChI Key"].map(normalize_upper)
    if "Synonyms" in vocab.columns:
        vocab["Synonyms"] = vocab["Synonyms"].map(normalize_text)
    else:
        vocab["Synonyms"] = ""

    inchikey_to_ids = {}
    name_to_ids = {}
    for _, row in vocab.iterrows():
        drugbank_id = row["DrugBank ID"]
        if not drugbank_id:
            continue

        if row["Standard InChI Key"]:
            inchikey_to_ids.setdefault(row["Standard InChI Key"], set()).add(drugbank_id)

        names = []
        if row["Common name"]:
            names.append(row["Common name"])
        names.extend(split_exact_aliases(row["Synonyms"]))
        for name in names:
            key = normalize_lower(name)
            if key:
                name_to_ids.setdefault(key, set()).add(drugbank_id)

    bridge_df = pd.read_csv(pubchem_drugbank_path, low_memory=False)
    bridge_df["drugbank_id"] = bridge_df["drugbank_id"].map(normalize_text)
    bridge_df["pubchem_Compound_CID"] = bridge_df["pubchem_Compound_CID"].map(normalize_text)
    bridge_df["name"] = bridge_df["name"].map(normalize_text)
    bridge_df["pubchem_drug_name"] = bridge_df["pubchem_drug_name"].map(normalize_text)
    drugbank_to_pubchem = (
        bridge_df.drop_duplicates(subset=["drugbank_id"])
        .set_index("drugbank_id")[["pubchem_Compound_CID", "name", "pubchem_drug_name"]]
        .to_dict(orient="index")
    )

    return inchikey_to_ids, name_to_ids, drugbank_to_pubchem


def fetch_pubchem_cids_from_api(namespace, value):
    """调用 PubChem PUG REST，只接受明确返回的 CID 列表。"""
    clean_value = normalize_text(value)
    if not clean_value:
        return set()

    cache_key = (namespace, clean_value)
    if cache_key in PUBCHEM_CACHE:
        return set(PUBCHEM_CACHE[cache_key])

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace}/{quote(clean_value, safe='')}/cids/TXT"
    try:
        response = requests.get(url, timeout=PUBCHEM_TIMEOUT)
    except requests.RequestException:
        return set()
    finally:
        time.sleep(PUBCHEM_SLEEP_SECONDS)

    if response.status_code != 200:
        return set()

    lines = [item.strip() for item in response.text.splitlines() if item.strip()]
    valid_cids = set()
    for item in lines:
        if item.isdigit() and item != "0":
            valid_cids.add(item)
    PUBCHEM_CACHE[cache_key] = sorted(valid_cids)
    return valid_cids


def choose_unique_value(candidates):
    """只有候选唯一时才接受。"""
    values = sorted({item for item in candidates if item})
    if len(values) == 1:
        return values[0]
    return ""


def choose_unique_from_source_map(source_map):
    """把每个来源的唯一结果再汇总成一个最终唯一值。"""
    unique_values = []
    for candidates in source_map.values():
        unique_value = choose_unique_value(candidates)
        if unique_value:
            unique_values.append(unique_value)
    return choose_unique_value(unique_values)


def resolve_pubchem_mapping(info, local_name_to_cids):
    """按结构优先、名称辅助的顺序获取 PubChem CID。"""
    structure_candidates = {}
    name_candidates = {}

    cmap_name = info["cmap_name"]
    if cmap_name:
        local_name_cids = local_name_to_cids.get(normalize_lower(cmap_name), set())
        if local_name_cids:
            name_candidates["local_pubchem_name_exact"] = set(local_name_cids)

    for alias in split_exact_aliases(info["compound_aliases"]):
        local_alias_cids = local_name_to_cids.get(normalize_lower(alias), set())
        if local_alias_cids:
            name_candidates[f"local_pubchem_alias_exact:{alias}"] = set(local_alias_cids)

    inchi_key = info["inchi_key"]
    if inchi_key:
        cids = fetch_pubchem_cids_from_api("inchikey", inchi_key)
        if cids:
            structure_candidates["api_inchikey_exact"] = cids

    # 结构证据还不能唯一落定时，再用名称 API 扩展候选。
    if not choose_unique_from_source_map(structure_candidates) and cmap_name:
        cids = fetch_pubchem_cids_from_api("name", cmap_name)
        if cids:
            name_candidates["api_name_exact"] = cids

    # InChIKey 已经能唯一确定时，不再重复请求 SMILES，减少全量构建时间。
    smiles = info["canonical_smiles"]
    if smiles and not choose_unique_from_source_map(structure_candidates):
        cids = fetch_pubchem_cids_from_api("smiles", smiles)
        if cids:
            structure_candidates["api_smiles_exact"] = cids

    final_structure_cid = choose_unique_from_source_map(structure_candidates)
    final_name_cid = choose_unique_from_source_map(name_candidates)

    detail = {
        "structure_sources": {
            source: choose_unique_value(cids)
            for source, cids in structure_candidates.items()
            if choose_unique_value(cids)
        },
        "name_sources": {
            source: choose_unique_value(cids)
            for source, cids in name_candidates.items()
            if choose_unique_value(cids)
        },
    }

    if final_structure_cid:
        status = "matched"
        final_cid = final_structure_cid
    elif structure_candidates:
        status = "conflict"
        final_cid = ""
    elif final_name_cid:
        status = "matched_by_name"
        final_cid = final_name_cid
    elif name_candidates:
        status = "conflict_by_name"
        final_cid = ""
    else:
        status = "unmatched"
        final_cid = ""

    if final_structure_cid and final_name_cid and final_structure_cid != final_name_cid:
        detail["name_conflict_with_structure"] = final_name_cid

    return final_cid, detail, status


def resolve_drugbank_mapping(info, inchikey_to_ids, name_to_ids):
    """按结构优先、名称辅助的顺序获取 DrugBank ID。"""
    structure_candidates = {}
    name_candidates = {}

    inchi_key = info["inchi_key"]
    if inchi_key:
        ids = inchikey_to_ids.get(inchi_key, set())
        if ids:
            structure_candidates["drugbank_inchikey_exact"] = set(ids)

    cmap_name = info["cmap_name"]
    if cmap_name:
        ids = name_to_ids.get(normalize_lower(cmap_name), set())
        if ids:
            name_candidates["drugbank_name_exact"] = set(ids)

    for alias in split_exact_aliases(info["compound_aliases"]):
        ids = name_to_ids.get(normalize_lower(alias), set())
        if ids:
            name_candidates[f"drugbank_alias_exact:{alias}"] = set(ids)

    final_structure_id = choose_unique_from_source_map(structure_candidates)
    final_name_id = choose_unique_from_source_map(name_candidates)

    detail = {
        "structure_sources": {
            source: choose_unique_value(ids)
            for source, ids in structure_candidates.items()
            if choose_unique_value(ids)
        },
        "name_sources": {
            source: choose_unique_value(ids)
            for source, ids in name_candidates.items()
            if choose_unique_value(ids)
        },
    }

    if final_structure_id:
        status = "matched"
        final_id = final_structure_id
    elif structure_candidates:
        status = "conflict"
        final_id = ""
    elif final_name_id:
        status = "matched_by_name"
        final_id = final_name_id
    elif name_candidates:
        status = "conflict_by_name"
        final_id = ""
    else:
        status = "unmatched"
        final_id = ""

    if final_structure_id and final_name_id and final_structure_id != final_name_id:
        detail["name_conflict_with_structure"] = final_name_id

    return final_id, detail, status


def bridge_pubchem_from_drugbank(drugbank_id, drugbank_to_pubchem):
    """最后用桥接表做 PubChem 补充和一致性校验。"""
    if not drugbank_id or drugbank_id not in drugbank_to_pubchem:
        return "", "", ""

    item = drugbank_to_pubchem[drugbank_id]
    cid = normalize_text(item["pubchem_Compound_CID"])
    if cid == "-":
        cid = ""
    return cid, normalize_text(item["name"]), normalize_text(item["pubchem_drug_name"])


def build_mapping_rows(
    current_drugs,
    base_info,
    local_name_to_cids,
    inchikey_to_ids,
    name_to_ids,
    drugbank_to_pubchem,
):
    """构建最终映射表。"""
    rows = []
    report = {
        "drug_count": len(current_drugs),
        "pubchem_direct_matched_count": 0,
        "drugbank_matched_count": 0,
        "bridge_pubchem_filled_count": 0,
        "final_pubchem_count": 0,
        "pubchem_conflict_count": 0,
        "drugbank_conflict_count": 0,
    }

    for pert_id in current_drugs:
        info = base_info.get(
            pert_id,
            {
                "cmap_name": "",
                "canonical_smiles": "",
                "inchi_key": "",
                "compound_aliases": "",
            },
        )

        pubchem_direct_cid, pubchem_detail, pubchem_status = resolve_pubchem_mapping(info, local_name_to_cids)
        drugbank_id, drugbank_detail, drugbank_status = resolve_drugbank_mapping(info, inchikey_to_ids, name_to_ids)
        bridge_pubchem_cid, bridge_drugbank_name, bridge_pubchem_name = bridge_pubchem_from_drugbank(
            drugbank_id,
            drugbank_to_pubchem,
        )

        final_pubchem_cid = pubchem_direct_cid
        final_pubchem_source = "direct_pubchem"
        bridge_status = ""

        if pubchem_direct_cid:
            if bridge_pubchem_cid:
                if pubchem_direct_cid == bridge_pubchem_cid:
                    bridge_status = "consistent"
                else:
                    bridge_status = "conflict"
            else:
                bridge_status = "no_bridge_pubchem"
        else:
            if bridge_pubchem_cid:
                final_pubchem_cid = bridge_pubchem_cid
                final_pubchem_source = "bridge_from_drugbank"
                bridge_status = "filled_from_bridge"
                report["bridge_pubchem_filled_count"] += 1
            else:
                final_pubchem_source = ""
                bridge_status = "unavailable"

        if pubchem_direct_cid:
            report["pubchem_direct_matched_count"] += 1
        if drugbank_id:
            report["drugbank_matched_count"] += 1
        if final_pubchem_cid:
            report["final_pubchem_count"] += 1
        if pubchem_status == "conflict":
            report["pubchem_conflict_count"] += 1
        if drugbank_status == "conflict":
            report["drugbank_conflict_count"] += 1

        rows.append(
            {
                "pert_id": pert_id,
                "cmap_name": info["cmap_name"],
                "canonical_smiles": info["canonical_smiles"],
                "inchi_key": info["inchi_key"],
                "compound_aliases": info["compound_aliases"],
                "pubchem_cid_direct": pubchem_direct_cid,
                "pubchem_direct_sources": json.dumps(pubchem_detail, ensure_ascii=False, sort_keys=True),
                "pubchem_direct_status": pubchem_status,
                "drugbank_id": drugbank_id,
                "drugbank_sources": json.dumps(drugbank_detail, ensure_ascii=False, sort_keys=True),
                "drugbank_status": drugbank_status,
                "bridge_pubchem_cid": bridge_pubchem_cid,
                "bridge_drugbank_name": bridge_drugbank_name,
                "bridge_pubchem_name": bridge_pubchem_name,
                "final_pubchem_cid": final_pubchem_cid,
                "final_pubchem_source": final_pubchem_source,
                "bridge_status": bridge_status,
            }
        )

    return pd.DataFrame(rows), report


def main():
    parser = argparse.ArgumentParser(description="构建当前训练 BRD 到 PubChem / DrugBank 的严格映射表")
    parser.add_argument(
        "--project_feature_dir",
        type=Path,
        default=Path("pythonPredict"),
        help="当前 pythonPredict 特征目录",
    )
    parser.add_argument(
        "--compoundinfo_path",
        type=Path,
        default=Path(r"D:\learning\buliangfanying\数据集\compoundinfo_beta.txt"),
        help="compoundinfo_beta.txt 路径",
    )
    parser.add_argument(
        "--drugbank_vocab_path",
        type=Path,
        default=Path(r"D:\learning\buliangfanying\数据集\drugbank vocabulary.csv"),
        help="drugbank vocabulary.csv 路径",
    )
    parser.add_argument(
        "--pubchem_drugbank_path",
        type=Path,
        default=Path(r"D:\learning\buliangfanying\数据集\drugbank2_pubchem_new\puchem_drugbank_new.csv"),
        help="puchem_drugbank_new.csv 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("pythonPredict/aligned_mapping"),
        help="映射表输出目录",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    current_drugs = load_current_drugs(args.project_feature_dir)
    base_info = load_base_info(current_drugs, args.compoundinfo_path)
    local_name_to_cids = build_local_pubchem_name_index(args.pubchem_drugbank_path)
    inchikey_to_ids, name_to_ids, drugbank_to_pubchem = build_drugbank_indexes(
        args.drugbank_vocab_path,
        args.pubchem_drugbank_path,
    )

    mapping_df, report = build_mapping_rows(
        current_drugs=current_drugs,
        base_info=base_info,
        local_name_to_cids=local_name_to_cids,
        inchikey_to_ids=inchikey_to_ids,
        name_to_ids=name_to_ids,
        drugbank_to_pubchem=drugbank_to_pubchem,
    )

    output_csv = args.output_dir / "brd_drugbank_pubchem_mapping.csv"
    mapping_df.to_csv(output_csv, index=False)

    with open(args.output_dir / "mapping_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"映射表输出: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
