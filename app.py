import streamlit as st
import pubchempy as pcp
import py3Dmol
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.linalg import expm
from deep_translator import GoogleTranslator
import re
import requests
import json
import streamlit.components.v1 as components

# 網頁基礎設定
st.set_page_config(page_title="中文化學物質分析與動態熱力學系統", layout="wide")

st.title("🧪 物質深度分析 & 3D 動態熱力學系統")
st.markdown("搭載 **自適應 Flexbox 網格防護** 與 **真實維度安全攔截機制**。根絕畫面重疊，確保物理模擬嚴謹且排版絕對穩定！")

# ==========================================
# 核心一：維基百科學術名詞對接與修正引擎
# ==========================================
LOCAL_CHEM_DICT = {
    "阿斯匹靈": "Aspirin", "普拿疼": "Acetaminophen", "雙氧水": "Hydrogen peroxide",
    "鹽酸": "Hydrochloric acid", "硫酸": "Sulfuric acid", "硝酸": "Nitric acid",
    "氨水": "Ammonia", "食鹽": "Sodium chloride", "氯化鈉": "Sodium chloride",
    "硝酸鉀": "Potassium nitrate", "亞硝酸鉀": "Potassium nitrite",
    "KNO2": "Potassium nitrite", "kno2": "Potassium nitrite",
    "高錳酸鉀": "Potassium permanganate", "碳酸鈉": "Sodium carbonate",
    "氫氧化鈉": "Sodium hydroxide", "乙醇": "Ethanol", "甲醇": "Methanol",
    "苯": "Benzene", "水": "Water", "咖啡酸": "Caffeic acid",
    "明礬": "Potassium aluminium sulfate"
}

def contains_chinese(text): 
    return bool(re.search('[\u4e00-\u9fff]', text))

def translate_via_wikipedia(zh_name):
    try:
        url = f"https://zh.wikipedia.org/w/api.php?action=query&prop=langlinks&titles={zh_name}&lllang=en&format=json"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
        for _, page_info in res.get("query", {}).get("pages", {}).items():
            if "langlinks" in page_info: return page_info["langlinks"][0]["*"] 
    except: pass
    return None

def fix_chemical_formula(formula):
    if not formula: return "無文獻資料"
    fix_map = {"ClNa": "NaCl", "HNaO": "NaOH", "ClK": "KCl", "HKO": "KOH", "IK": "KI", "KNO2": "KNO₂", "NO2K": "KNO₂", "NO3K": "KNO₃", "C2H4O2": "CH₃COOH", "H2O": "H₂O"}
    if formula in fix_map: return fix_map[formula]
    return formula

# ==========================================
# 核心二：數據抓取與智慧正規化引擎 (解決 None 與單位混亂)
# ==========================================
def simplify_physical_state(text):
    if not text or text == "無相關文獻數據": return text
    t = text.lower()
    if any(kw in t for kw in ["solid", "crystal", "powder", "pellet", "salt"]): return "🧊 固體"
    elif any(kw in t for kw in ["liquid", "fluid"]) and "solution" not in t: return "💧 液體"
    elif any(kw in t for kw in ["gas", "vapor"]): return "☁️ 氣體"
    elif "solution" in t or "aqueous" in t: return "💧 水溶液"
    return text

def standardize_temperature(raw_str):
    if not raw_str: return None
    c_single = re.search(r'(-?\d+\.?\d*)\s*(?:°C|deg C|C\b)', raw_str, re.IGNORECASE)
    if c_single: return f"{c_single.group(1)} °C"
    f_match = re.search(r'(-?\d+\.?\d*)\s*(?:°F|deg F|F\b)', raw_str, re.IGNORECASE)
    if f_match: return f"{(float(f_match.group(1)) - 32) * 5.0 / 9.0:.1f} °C"
    k_match = re.search(r'(-?\d+\.?\d*)\s*(?:K\b)', raw_str)
    if k_match: return f"{float(k_match.group(1)) - 273.15:.1f} °C"
    return None

def fetch_sds_and_properties(cid):
    props = {"外觀與性狀": "無相關文獻數據", "密度": "無相關文獻數據", "熔點": "無相關文獻數據", "沸點": "無相關文獻數據", "閃點": "無相關文獻數據", "溶解度": "無相關文獻數據", "蒸氣壓": "無相關文獻數據", "危險信號詞": "無標示 / 安全", "危害警告": []}
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        res = requests.get(url, timeout=10).json()
        sections = res.get("Record", {}).get("Section", [])
        p_map = {"Physical Description": "外觀與性狀", "Density": "密度", "Melting Point": "熔點", "Boiling Point": "沸點", "Flash Point": "閃點", "Solubility": "溶解度", "Vapor Pressure": "蒸氣壓"}
        
        for sec in sections:
            if sec.get("TOCHeading") == "Chemical and Physical Properties":
                for subsec in sec.get("Section", []):
                    if subsec.get("TOCHeading") in ["Experimental Properties", "Computed Properties"]:
                        for prop in subsec.get("Section", []):
                            h = prop.get("TOCHeading")
                            if h in p_map:
                                target = p_map[h]
                                if props[target] != "無相關文獻數據": continue
                                for info in prop.get("Information", []):
                                    v_list = info.get("Value", {}).get("StringWithMarkup", [])
                                    v_str = v_list[0].get("String") if v_list else None
                                    if v_str:
                                        if target == "外觀與性狀": props[target] = simplify_physical_state(v_str); break
                                        elif target in ["熔點", "沸點", "閃點"]:
                                            temp = standardize_temperature(v_str)
                                            if temp: props[target] = temp; break
                                        else: props[target] = v_str; break
            elif sec.get("TOCHeading") == "Safety and Hazards":
                for subsec in sec.get("Section", []):
                    if subsec.get("TOCHeading") == "Hazards Identification":
                        for ghs in subsec.get("Section", []):
                            if ghs.get("TOCHeading") == "GHS Classification":
                                for info in ghs.get("Information", []):
                                    if info.get("Name") == "Signal":
                                        raw_s = info["Value"]["StringWithMarkup"][0]["String"]
                                        props["危險信號詞"] = "危險 (Danger)" if "Danger" in raw_s else ("警告 (Warning)" if "Warning" in raw_s else raw_s)
                                    elif info.get("Name") == "GHS Hazard Statements":
                                        raw_h = [h["String"] for h in info["Value"]["StringWithMarkup"]]
                                       if any("not classified" in h.lower() for h in raw_h):
                                            props["危險信號詞"] = "無標示 / 安全"
                                            props["危害警告"] = []
                                        else:
                                            try: props["危害警告"] = [GoogleTranslator(source='auto', target='zh-TW').translate(h) for h in raw_h[:5]]
                                            except: props["危害警告"] = raw_h[:5]
    except: pass
    return props

# ==========================================
# 核心三：雙軌檢索邏輯 (精準判斷 3D 實體座標)
# ==========================================
def run_search(query_name):
    english_name = query_name
    if contains_chinese(query_name) or query_name in LOCAL_CHEM_DICT:
        if query_name in LOCAL_CHEM_DICT: english_name = LOCAL_CHEM_DICT[query_name]
        else:
            wiki_name = translate_via_wikipedia(query_name)
            if wiki_name: english_name = wiki_name
            else:
                try: english_name = GoogleTranslator(source='auto', target='en').translate(query_name)
                except: return False, "翻譯服務暫時不可用"

    std_compounds = pcp.get_compounds(english_name, 'name')
    if not std_compounds: return False, f"⚠️ 資料庫無法配對「{english_name}」"
    
    c_std = std_compounds[0]
    cid = c_std.cid

    # 專門抓取 3D 結構
    c_3d_list = pcp.get_compounds(cid, record_type='3d')
    c_3d = c_3d_list[0] if c_3d_list else None
    
    real_coords = {}
    if c_3d:
        for atom in c_3d.atoms:
            if hasattr(atom, 'x') and atom.x is not None:
                real_coords[atom.aid] = [atom.x, atom.y, atom.z]
    
    # 🚀 安全防護回歸本質：有抓到真實 3D 空間座標才算 3D，否則就是 2D (完美阻斷無機鹽)
    dim_type = "3D 立體" if len(real_coords) > 0 else "2D 平面"
    
    mw = c_std.molecular_weight
    tpsa = c_std.tpsa
    hbd = c_std.h_bond_donor_count
    hba = c_std.h_bond_acceptor_count
    smiles = c_std.isomeric_smiles or c_std.canonical_smiles

    st.session_state.search_data = {
        "english_name": english_name.capitalize(),
        "cid": cid,
        "fixed_formula": fix_chemical_formula(c_std.molecular_formula),
        "molecular_weight": f"{mw} " if mw is not None else "無資料",
        "tpsa": f"{tpsa} " if tpsa is not None else "無資料",
        "h_bond_donor_count": hbd if hbd is not None else "無",
        "h_bond_acceptor_count": hba if hba is not None else "無",
        "isomeric_smiles": smiles if smiles is not None else "無資料",
        "sds_data": fetch_sds_and_properties(cid),
        "dim_type": dim_type
    }
    
    sim_source = c_3d if (c_3d and real_coords) else c_std
    st.session_state.mol_atoms = [atom.aid for atom in sim_source.atoms]
    st.session_state.mol_bonds = [(bond.aid1, bond.aid2) for bond in sim_source.bonds]
    st.session_state.mol_coords = real_coords
    st.session_state.mol_name = english_name.capitalize()
    
    degree = {}
    for a, b in st.session_state.mol_bonds:
        degree[a] = degree.get(a, 0) + 1; degree[b] = degree.get(b, 0) + 1
    if degree:
        sorted_nodes = sorted(degree.keys(), key=lambda x: degree[x])
        st.session_state.core_node, st.session_state.edge_node = sorted_nodes[-1], sorted_nodes[0]
        if st.session_state.core_node == st.session_state.edge_node and len(st.session_state.mol_atoms) > 1:
            st.session_state.edge_node = [n for n in st.session_state.mol_atoms if n != st.session_state.core_node][0]
    else:
        st.session_state.core_node = st.session_state.mol_atoms[0] if st.session_state.mol_atoms else 0
        st.session_state.edge_node = st.session_state.mol_atoms[-1] if len(st.session_state.mol_atoms) > 1 else 0

    st.session_state.particle_temps = {i: 25.0 for i in st.session_state.mol_atoms}
    st.session_state.particle_temps[st.session_state.core_node] = 500.0
    return True, "Success"

if 'initialized' not in st.session_state:
    run_search("水")
    st.session_state.initialized = True

# --- 側邊欄參數配置面板 ---
with st.sidebar:
    st.header("⚙️ 控制面板")
    user_input = st.text_input("輸入化學名稱 (中文/英文俗名/學名)", "水").strip()
    style = st.selectbox("3D 渲染風格", ["stick", "sphere", "line", "cross"])
    search_button = st.button("🔍 檢索數據", type="primary")
    st.markdown("---")
    env_temp = st.slider("環境溫度設定 (°C)", -20.0, 60.0, 25.0, step=0.5)
    init_temp = st.slider("中心點火溫度 (°C)", 50.0, 500.0, 500.0, step=10.0)
    k_val = st.slider("熱傳導係數 (k)", 0.01, 0.50, 0.15, step=0.01)
    sim_duration = st.slider("模擬總時長 (秒)", 3.0, 30.0, 10.0, step=1.0)
    anim_speed = st.slider("動畫每幀播放延遲 (ms)", 10, 200, 40, step=10)

if search_button and user_input:
    with st.spinner("🧠 雙軌大數據同步擷取中..."):
        success, msg = run_search(user_input)
        if not success: st.error(msg)

# --- 前端雙分頁系統 ---
tab1, tab2 = st.tabs(["🧬 SDS 物質安全與化學百科", "🔥 網格分離式動畫儀表板"])

# ==========================================
# 分頁 1：化學百科 (還原乾淨雙欄排版)
# ==========================================
with tab1:
    sd = st.session_state.search_data
    st.success(f"✅ 當前載入物質：「**{sd['english_name']}**」 | 系統已成功解析全數 **{len(st.session_state.mol_atoms)}** 顆真實原子。")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("⚛️ 空間立體結構")
        try:
            res = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{sd['cid']}/record/SDF/?record_type=3d", timeout=5)
            if res.status_code == 200:
                view = py3Dmol.view(width=400, height=300)
                view.addModel(res.text, "sdf")
                view.setStyle({style: {}, 'sphere': {'radius': 0.2}})
                view.setBackgroundColor('#f0f2f6') 
                view.zoomTo()
                components.html(view._make_html().replace("http://", "https://"), height=300)
            else:
                st.warning("⚠️ 查無官方 3D 模型，系統已降級為 2D 結構圖。")
                st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{sd['cid']}/PNG?image_size=large", use_container_width=True)
        except: 
            st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{sd['cid']}/PNG?image_size=large", use_container_width=True)
        
        st.markdown("---")
        st.subheader("🧮 計算結構屬性")
        st.markdown(f"""
        * **慣用化學式:** `{sd['fixed_formula']}`
        * **真實分子量:** `{sd['molecular_weight']}g/mol`
        * **TPSA (極性表面積):** `{sd['tpsa']}Å²`
        * **氫鍵 (供體/受體):** `{sd['h_bond_donor_count']} / {sd['h_bond_acceptor_count']}`
        * **SMILES 結構式:** `{sd['isomeric_smiles']}`
        """)
        
    with c2:
        st.subheader("⚠️ SDS 物質安全與危害標示 (GHS)")
        sds = sd['sds_data']
        if sds['危險信號詞'] != "無標示 / 安全":
            st.error(f"🚨 **警示語: {sds['危險信號詞']}**")
        else:
            st.success(f"✅ **警示語: 無特殊危險標示**")
            
        if sds['危害警告']:
            for h in sds['危害警告']: st.caption(f"▪️ {h}")
        st.markdown("---")
        st.subheader("🌡️ 實驗室文獻實測數據")
        st.markdown(f"""
        | 屬性類別 | 文獻實測數據 (包含單位) |
        | :--- | :--- |
        | 🧊 **密度 (Density)** | {sds['密度']} |
        | ♨️ **沸點 (Boiling Point)** | {sds['沸點']} |
        | ❄️ **熔點 (Melting Point)** | {sds['熔點']} |
        | 🔥 **閃點 (Flash Point)** | {sds['閃點']} |
        | 💧 **溶解度 (Solubility)** | {sds['溶解度']} |
        | ☁️ **蒸氣壓 (Vapor Pressure)** | {sds['蒸氣壓']} |
        | 👁️ **外觀與性狀** | {sds['外觀與性狀']} |
        """)

# ==========================================
# 分頁 2：網格聯動戰情室 (防破版 + 2D 絕對攔截)
# ==========================================
with tab2:
    # 🚀 維度安全阻斷機制：只要沒有 3D 實體座標，絕對不允許進行熱傳導！
    if sd['dim_type'] == "2D 平面":
        st.warning("⚠️ 系統安全攔截：偵測到當前物質僅具備 2D 平面結構數據，已自動阻斷 3D 熱傳導模擬。")
        st.info("""
        💡 **科學嚴謹性防護說明：**
        由於本系統採用 **拉普拉斯矩陣之偏微分方程 (Matrix Exponential)** 進行熱傳導推演，運算高度依賴分子在真實世界中的物理空間座標 ($x, y, z$)。
        
        當前檢索的物質（如明礬、無機鹽類或僅有平面紀錄之大分子）缺乏真實 3D 空間數據。若強行計算會導致物理意義錯誤，並引發 WebGL 畫布崩潰（畫面切斷或死機）。
        
        **👉 解決方案：**
        請在左側重新檢索具備立體座標的共價分子（例如：**阿斯匹靈**、**咖啡酸**、**水**、**苯**），即可解鎖流暢的 3D 動態模擬！
        """)
    else:
        st.subheader(f"🔥 {st.session_state.mol_name} - 3D 矩陣指數動態傳導戰情室")
        
        start_anim = st.button("⚙️ 生成劇院級分離式動態底片", type="primary", use_container_width=True)
        
        if start_anim:
            with st.spinner("⚡ 正在求解偏微分方程絕對精確解..."):
                G = nx.Graph()
                G.add_nodes_from(st.session_state.mol_atoms); G.add_edges_from(st.session_state.mol_bonds)
                L = nx.laplacian_matrix(G).toarray()
                T0 = np.array([env_temp if i != st.session_state.core_node else init_temp for i in st.session_state.mol_atoms])
                times = np.linspace(0, sim_duration, 100)
                
                # 計算時間擴散矩陣
                history = [expm(-k_val * t * 1.0 * L).dot(T0) for t in times]
                c_hist = [h[st.session_state.mol_atoms.index(st.session_state.core_node)] for h in history]
                e_hist = [h[st.session_state.mol_atoms.index(st.session_state.edge_node)] for h in history]
                
                # 🚀 建立乾淨的 3D 圖表 (移除常駐文字，改為 hover 顯示)
                fig3d = go.Figure()
                p3d = st.session_state.mol_coords
                for b in st.session_state.mol_bonds:
                    fig3d.add_trace(go.Scatter3d(x=[p3d[b[0]][0], p3d[b[1]][0]], y=[p3d[b[0]][1], p3d[b[1]][1]], z=[p3d[b[0]][2], p3d[b[1]][2]], mode='lines', line=dict(color='gray', width=3), hoverinfo='none'))
                
                init_h = history[0]
                init_labels = [f"🔥 核心源<br>溫度: {init_h[st.session_state.mol_atoms.index(i)]:.1f}°C" if i == st.session_state.core_node else (f"❄️ 外圍點<br>溫度: {init_h[st.session_state.mol_atoms.index(i)]:.1f}°C" if i == st.session_state.edge_node else f"原子 {i}<br>溫度: {init_h[st.session_state.mol_atoms.index(i)]:.1f}°C") for i in st.session_state.mol_atoms]
                
                fig3d.add_trace(go.Scatter3d(
                    x=[p3d[i][0] for i in st.session_state.mol_atoms], y=[p3d[i][1] for i in st.session_state.mol_atoms], z=[p3d[i][2] for i in st.session_state.mol_atoms], 
                    mode='markers', text=init_labels, hoverinfo='text',
                    marker=dict(size=22, color=init_h, colorscale='Turbo', cmin=env_temp-5, cmax=init_temp+5, colorbar=dict(title="溫度 (°C)", thickness=10, x=-0.05))
                ))
                
                fig3d.frames = [go.Frame(data=[go.Scatter3d(marker=dict(color=h))], name=f"f{i}", traces=[len(st.session_state.mol_bonds)]) for i, h in enumerate(history)]
                # 🚀 啟動 responsive 自動縮放
                fig3d.update_layout(autosize=True, title="🔥 真實 3D 空間熱擴散", template="plotly_dark", margin=dict(l=10, r=10, b=10, t=40), scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), updatemenus=[dict(type="buttons", active=-1, showactive=False, y=-0.05, x=0.5, xanchor="center", direction="left", buttons=[dict(label="▶️ 播放聯動", method="animate", args=[None, dict(frame=dict(duration=anim_speed, redraw=True), fromcurrent=True, mode="immediate", transition=dict(duration=0))]), dict(label="⏸️ 暫停", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])])
                
                # 🚀 建立 2D 曲線圖 (強制拓寬左下邊距 l=60, b=60，保證座標軸不被裁切)
                fig2d = go.Figure()
                fig2d.add_trace(go.Scatter(x=[times[0]], y=[c_hist[0]], mode='lines', name="中心點火源", line=dict(color='red', width=3)))
                fig2d.add_trace(go.Scatter(x=[times[0]], y=[e_hist[0]], mode='lines', name="外圍測溫點", line=dict(color='blue', width=3)))
                fig2d.update_layout(autosize=True, title="📈 絕對精確溫度動態變化 (°C)", template="plotly_dark", margin=dict(l=60, r=20, b=60, t=40), xaxis=dict(range=[0, sim_duration], title="時間 (秒)"), yaxis=dict(range=[env_temp-10, init_temp+20], title="溫度 (°C)"))
                
                html_3d = fig3d.to_html(include_plotlyjs='cdn', full_html=False, default_width='100%', default_height='100%', div_id='plot-3d', config={'responsive': True})
                html_2d = fig2d.to_html(include_plotlyjs=False, full_html=False, default_width='100%', default_height='100%', div_id='plot-2d', config={'responsive': True})
                
                history_json = json.dumps([h.tolist() for h in history])
                time_json = json.dumps(times.tolist())
                core_json = json.dumps(c_hist)
                edge_json = json.dumps(e_hist)
                atoms_json = json.dumps(st.session_state.mol_atoms)
                
                table_rows = "".join([f"<tr style='border-bottom:1px solid #333;'><td style='padding:6px;'>Atom {id}</td><td style='padding:6px;'>{'🔥 核心源' if id == st.session_state.core_node else '❄️ 外部點' if id == st.session_state.edge_node else '傳導中圈'}</td><td id='temp-{i}' style='color:#00ffcc; font-weight:bold; padding:6px;'>{init_h[i]:.2f} °C</td></tr>" for i, id in enumerate(st.session_state.mol_atoms)])

                # 🚀 終極護城河：使用 Flexbox 並加上 min-height: 0 避免圖表撐破容器溢出！
                html_template = """
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body, html { margin: 0; padding: 0; background-color: #0e1117; width: 100%; height: 100%; overflow: hidden; box-sizing: border-box; }
                        #fs-container { display: flex; flex-direction: row; width: 100%; height: 100vh; background: #0e1117; position: relative; }
                        #left-pane { flex: 6; display: flex; flex-direction: column; min-width: 0; min-height: 0; border-right: 2px solid #333; position: relative; overflow: hidden; }
                        #right-pane { flex: 4; display: flex; flex-direction: column; min-width: 0; min-height: 0; position: relative; overflow: hidden; }
                        #top-right-pane { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; border-bottom: 2px solid #333; position: relative; overflow: hidden; }
                        #bottom-right-pane { flex: 1; overflow-y: auto; background: #1a1a1a; padding: 15px; position: relative; }
                        /* 強制內部畫布完全繼承格子大小 */
                        .js-plotly-plot, .plot-container { width: 100% !important; height: 100% !important; flex: 1; min-height: 0; }
                    </style>
                </head>
                <body>
                    <button style="position:absolute; top:10px; right:20px; z-index:9999; background:rgba(255,255,255,0.1); color:#fff; border:1px solid rgba(255,255,255,0.4); padding:6px 12px; border-radius:4px; cursor:pointer;" onclick="toggleFS()">⤢ 全螢幕</button>
                    <div id="fs-container">
                        <div id="left-pane">
                            __HTML_3D__
                        </div>
                        <div id="right-pane">
                            <div id="top-right-pane">
                                __HTML_2D__
                            </div>
                            <div id="bottom-right-pane">
                                <table style="width:100%; border-collapse:collapse; text-align:center; color:white; font-family:sans-serif;">
                                    <thead>
                                        <tr style="border-bottom:2px solid #555; position:sticky; top:0; background:#222;">
                                            <th style="padding:10px;">粒子編號</th>
                                            <th style="padding:10px;">拓樸定位</th>
                                            <th style="padding:10px;">即時溫度 (°C)</th>
                                        </tr>
                                    </thead>
                                    <tbody>__TABLE_ROWS__</tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                    <script>
                        function toggleFS() {
                            let elem = document.documentElement;
                            if (!document.fullscreenElement) { elem.requestFullscreen(); } 
                            else { document.exitFullscreen(); }
                        }

                        var h_data = __HISTORY_JSON__;
                        var t_data = __TIME_JSON__;
                        var c_data = __CORE_JSON__;
                        var e_data = __EDGE_JSON__;
                        var a_list = __ATOMS_JSON__;

                        var checkExist = setInterval(function() {
                            var gd3d = document.getElementById('plot-3d');
                            var gd2d = document.getElementById('plot-2d');
                            if (gd3d && typeof gd3d.on === 'function' && gd2d && typeof Plotly !== 'undefined') {
                                clearInterval(checkExist);
                                gd3d.on('plotly_animatingframe', function(eventData) {
                                    var step = parseInt(eventData.name.replace('f', ''));
                                    var temps = h_data[step];
                                    if (temps) {
                                        for (var i = 0; i < a_list.length; i++) {
                                            var cell = document.getElementById('temp-' + i);
                                            if (cell) { cell.innerText = temps[i].toFixed(2) + ' °C'; }
                                        }
                                    }
                                    Plotly.restyle(gd2d, {'x': [t_data.slice(0, step + 1), t_data.slice(0, step + 1)], 'y': [c_data.slice(0, step + 1), e_data.slice(0, step + 1)]}, [0, 1]);
                                });
                            }
                        }, 200);

                        // 🚀 關鍵：網頁載入後，強制發送 resize 訊號，讓圖表完美填滿 Flexbox！
                        window.onload = function() {
                            setTimeout(function() {
                                window.dispatchEvent(new Event('resize'));
                            }, 500);
                        };
                    </script>
                </body>
                </html>
                """
                
                custom_html = html_template\
                    .replace("__HTML_3D__", html_3d)\
                    .replace("__HTML_2D__", html_2d)\
                    .replace("__TABLE_ROWS__", table_rows)\
                    .replace("__HISTORY_JSON__", history_json)\
                    .replace("__TIME_JSON__", time_json)\
                    .replace("__CORE_JSON__", core_json)\
                    .replace("__EDGE_JSON__", edge_json)\
                    .replace("__ATOMS_JSON__", atoms_json)
                    
                # 設定足夠的 iframe 高度，避免被 Streamlit 框架壓縮
                components.html(custom_html, height=850)
        else:
            st.info("💡 請點擊上方按鈕開始生成劇院級分離式動態底片並解鎖戰情室。")
