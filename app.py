"""
大语言模型综合能力测评平台
================================
用于测试和比较不同大语言模型在各维度上的表现
"""

import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(
    page_title="LLM 测评平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono&display=swap');
    
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #f59e0b;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border-color: #334155;
        --success-color: #22c55e;
        --warning-color: #eab308;
        --error-color: #ef4444;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Noto Sans SC', sans-serif !important;
        color: var(--text-primary) !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        padding: 1rem 0;
    }
    
    .question-card {
        background: linear-gradient(145deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .question-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .question-id {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .dimension-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .dim-logic { background: #3b82f6; color: white; }
    .dim-coding { background: #22c55e; color: white; }
    .dim-language { background: #f59e0b; color: white; }
    .dim-tool_use { background: #8b5cf6; color: white; }
    .dim-safety { background: #ef4444; color: white; }
    
    .model-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        border-color: #6366f1;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
    }
    
    .copy-btn {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    .copy-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    .stats-card {
        background: linear-gradient(145deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stats-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    .sidebar .stSelectbox label {
        color: var(--text-primary) !important;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background: #1e293b !important;
        padding: 0.125rem 0.375rem !important;
        border-radius: 4px !important;
    }
    
    .stTextArea textarea {
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: 'Noto Sans SC', sans-serif !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    .success-message {
        background: linear-gradient(135deg, #166534, #15803d);
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 1rem;
        color: white;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
    }
    
    .answer-preview {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        max-height: 200px;
        overflow-y: auto;
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .score-5 { background: #22c55e; color: white; }
    .score-4 { background: #84cc16; color: white; }
    .score-3 { background: #eab308; color: black; }
    .score-2 { background: #f97316; color: white; }
    .score-1 { background: #ef4444; color: white; }
    .score-0 { background: #64748b; color: white; }
    
    .leaderboard-row {
        background: linear-gradient(145deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .rank-1 { border-left: 4px solid #ffd700; }
    .rank-2 { border-left: 4px solid #c0c0c0; }
    .rank-3 { border-left: 4px solid #cd7f32; }
</style>
""", unsafe_allow_html=True)

# 数据文件路径
QUESTIONS_FILE = "questions.json"
MODELS_FILE = "models.json"
ANSWERS_FILE = "answers.json"
SCORING_RUBRIC_FILE = "scoring_rubric.json"

# 维度名称映射
DIMENSION_NAMES = {
    "logic": "逻辑推理与数学",
    "coding": "代码与技术能力",
    "language": "语言理解与创作",
    "tool_use": "工具调用与格式化",
    "safety": "安全、伦理与幻觉"
}

DIMENSION_ICONS = {
    "logic": "🧮",
    "coding": "💻",
    "language": "✍️",
    "tool_use": "🔧",
    "safety": "🛡️"
}

# 每题满分
SCORE_PER_QUESTION = 5
TOTAL_QUESTIONS = 20
MAX_TOTAL_SCORE = SCORE_PER_QUESTION * TOTAL_QUESTIONS  # 100分


def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        st.error(f"JSON文件格式错误: {file_path}")
        return None


def save_json(file_path, data):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_answers():
    """加载答案数据"""
    data = load_json(ANSWERS_FILE)
    if data is None:
        data = {
            "meta": {
                "title": "模型回答记录",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "answers": []
        }
        save_json(ANSWERS_FILE, data)
    return data


def save_answer(question_id, model_id, answer_text):
    """保存单个答案"""
    data = load_answers()
    
    # 查找是否已有该记录
    existing_idx = None
    for idx, ans in enumerate(data["answers"]):
        if ans["question_id"] == question_id and ans["model_id"] == model_id:
            existing_idx = idx
            break
    
    answer_record = {
        "question_id": question_id,
        "model_id": model_id,
        "answer": answer_text,
        "timestamp": datetime.now().isoformat(),
        "score": None,
        "comment": None
    }
    
    if existing_idx is not None:
        # 保留原有分数
        if data["answers"][existing_idx].get("score") is not None:
            answer_record["score"] = data["answers"][existing_idx]["score"]
            answer_record["comment"] = data["answers"][existing_idx].get("comment")
        data["answers"][existing_idx] = answer_record
    else:
        data["answers"].append(answer_record)
    
    data["meta"]["last_updated"] = datetime.now().isoformat()
    save_json(ANSWERS_FILE, data)
    return True


def save_score(question_id, model_id, score, comment=None):
    """保存评分"""
    data = load_answers()
    
    for ans in data["answers"]:
        if ans["question_id"] == question_id and ans["model_id"] == model_id:
            ans["score"] = score
            ans["comment"] = comment
            ans["scored_at"] = datetime.now().isoformat()
            break
    
    data["meta"]["last_updated"] = datetime.now().isoformat()
    save_json(ANSWERS_FILE, data)
    return True


def get_answer(question_id, model_id):
    """获取指定问题和模型的答案"""
    data = load_answers()
    for ans in data["answers"]:
        if ans["question_id"] == question_id and ans["model_id"] == model_id:
            return ans
    return None


def get_model_scores(model_id):
    """获取模型的所有评分"""
    data = load_answers()
    scores = {}
    for ans in data["answers"]:
        if ans["model_id"] == model_id and ans.get("score") is not None:
            scores[ans["question_id"]] = ans["score"]
    return scores


def get_all_scores():
    """获取所有评分数据"""
    data = load_answers()
    questions_data = load_json(QUESTIONS_FILE)
    models_data = load_json(MODELS_FILE)
    
    if not questions_data or not models_data:
        return {}
    
    scores = {}
    for model in models_data["models"]:
        model_id = model["id"]
        scores[model_id] = {
            "name": model["name"],
            "icon": model["icon"],
            "scores": {},
            "total": 0,
            "answered": 0,
            "scored": 0
        }
        
        for ans in data["answers"]:
            if ans["model_id"] == model_id:
                scores[model_id]["answered"] += 1
                if ans.get("score") is not None:
                    scores[model_id]["scores"][ans["question_id"]] = ans["score"]
                    scores[model_id]["total"] += ans["score"]
                    scores[model_id]["scored"] += 1
    
    return scores


def get_statistics():
    """获取统计数据"""
    questions_data = load_json(QUESTIONS_FILE)
    models_data = load_json(MODELS_FILE)
    answers_data = load_answers()
    
    total_questions = len(questions_data["questions"]) if questions_data else 0
    total_models = len(models_data["models"]) if models_data else 0
    total_answers = len(answers_data["answers"])
    total_possible = total_questions * total_models
    completion_rate = (total_answers / total_possible * 100) if total_possible > 0 else 0
    
    # 统计已评分数量
    scored_count = sum(1 for ans in answers_data["answers"] if ans.get("score") is not None)
    scoring_rate = (scored_count / total_answers * 100) if total_answers > 0 else 0
    
    return {
        "total_questions": total_questions,
        "total_models": total_models,
        "total_answers": total_answers,
        "completion_rate": completion_rate,
        "scored_count": scored_count,
        "scoring_rate": scoring_rate
    }


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## 🧪 LLM 测评平台")
        st.markdown("---")
        
        # 导航
        page = st.radio(
            "📍 导航",
            ["🏠 首页", "📝 题目测评", "⭐ 评分打分", "📊 结果展示", "📋 数据查看", "⚙️ 设置"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 统计信息
        stats = get_statistics()
        st.markdown("### 📈 统计概览")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("题目数", stats["total_questions"])
            st.metric("已答题", stats["total_answers"])
        with col2:
            st.metric("模型数", stats["total_models"])
            st.metric("已评分", stats["scored_count"])
        
        st.markdown("---")
        st.markdown("### 🔗 模型链接")
        models_data = load_json(MODELS_FILE)
        if models_data:
            for model in models_data["models"][:5]:
                st.markdown(f"{model['icon']} [{model['name']}]({model['url']})")
        
        return page


def render_home():
    """渲染首页"""
    st.markdown('<h1 class="main-title">🧪 大语言模型综合能力测评平台</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; margin-bottom: 2rem;">
        测试并比较不同大语言模型在逻辑推理、代码能力、语言创作、工具调用和安全性等维度的表现
    </div>
    """, unsafe_allow_html=True)
    
    # 统计卡片
    stats = get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats['total_questions']}</div>
            <div class="stats-label">📋 测试题目</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats['total_models']}</div>
            <div class="stats-label">🤖 参评模型</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats['total_answers']}</div>
            <div class="stats-label">✅ 已收集答案</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats['scored_count']}</div>
            <div class="stats-label">⭐ 已评分</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 评分标准说明
    st.markdown("### 📐 评分标准")
    st.markdown("""
    | 分数 | 等级 | 描述 |
    |------|------|------|
    | **5分** | 完美 | 完整正确，逻辑清晰，有深度分析 |
    | **4分** | 优秀 | 基本正确，有小瑕疵，分析较完整 |
    | **3分** | 良好 | 主要部分正确，有明显遗漏或小错误 |
    | **2分** | 及格 | 部分正确，有较大错误或重要遗漏 |
    | **1分** | 不及格 | 仅少部分正确，大部分错误 |
    | **0分** | 错误 | 完全错误或未作答 |
    """)
    
    st.info(f"📊 **总分计算**: 每题满分 {SCORE_PER_QUESTION} 分 × {TOTAL_QUESTIONS} 题 = **{MAX_TOTAL_SCORE} 分**")
    
    st.markdown("---")
    
    # 维度介绍
    st.markdown("### 📐 测评维度")
    
    questions_data = load_json(QUESTIONS_FILE)
    if questions_data:
        cols = st.columns(5)
        for idx, dim in enumerate(questions_data["meta"]["dimensions"]):
            with cols[idx]:
                st.markdown(f"""
                <div class="stats-card">
                    <div style="font-size: 2rem;">{DIMENSION_ICONS.get(dim['id'], '📌')}</div>
                    <div style="font-weight: 600; color: #f1f5f9; margin: 0.5rem 0;">{dim['name']}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">{len(dim['questions'])} 道题</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 使用说明
    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. **📝 题目测评**: 复制题目，前往模型官网获取答案，粘贴回平台
    2. **⭐ 评分打分**: 根据评分标准为每个答案打分（0-5分）
    3. **📊 结果展示**: 查看排行榜、雷达图、维度对比等可视化分析
    4. **📋 数据查看**: 导出数据，查看答案对比
    """)


def render_questions():
    """渲染题目测评页面"""
    st.markdown('<h1 class="main-title">📝 题目测评</h1>', unsafe_allow_html=True)
    
    questions_data = load_json(QUESTIONS_FILE)
    models_data = load_json(MODELS_FILE)
    
    if not questions_data or not models_data:
        st.error("无法加载数据文件")
        return
    
    # 筛选器
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dimension_options = ["全部"] + [d["name"] for d in questions_data["meta"]["dimensions"]]
        selected_dimension = st.selectbox("🏷️ 选择维度", dimension_options)
    
    with col2:
        model_options = ["全部"] + [m["name"] for m in models_data["models"]]
        selected_model = st.selectbox("🤖 选择模型", model_options)
    
    with col3:
        question_ids = ["全部"] + [str(q["id"]) for q in questions_data["questions"]]
        selected_question = st.selectbox("📋 选择题号", question_ids)
    
    st.markdown("---")
    
    # 过滤题目
    filtered_questions = questions_data["questions"]
    
    if selected_dimension != "全部":
        dim_id = None
        for d in questions_data["meta"]["dimensions"]:
            if d["name"] == selected_dimension:
                dim_id = d["id"]
                break
        if dim_id:
            filtered_questions = [q for q in filtered_questions if q["dimension"] == dim_id]
    
    if selected_question != "全部":
        filtered_questions = [q for q in filtered_questions if str(q["id"]) == selected_question]
    
    # 显示题目
    for question in filtered_questions:
        dim_class = f"dim-{question['dimension']}"
        dim_name = DIMENSION_NAMES.get(question['dimension'], question['dimension'])
        dim_icon = DIMENSION_ICONS.get(question['dimension'], '📌')
        
        with st.expander(f"**第 {question['id']} 题**: {question['title']}", expanded=(selected_question != "全部")):
            # 题目信息
            st.markdown(f"""
            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
                <span class="dimension-badge {dim_class}">{dim_icon} {dim_name}</span>
                <span class="dimension-badge" style="background: #475569;">难度: {question.get('difficulty', 'N/A')}</span>
                <span class="dimension-badge" style="background: #475569;">类型: {question.get('type', 'N/A')}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # 题目内容
            st.markdown("#### 📋 题目内容")
            st.markdown(f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                <pre style="white-space: pre-wrap; word-wrap: break-word; color: #e2e8f0; margin: 0; font-family: 'Noto Sans SC', sans-serif;">{question['content']}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            # 复制按钮
            if st.button(f"📋 复制题目到剪贴板", key=f"copy_{question['id']}"):
                st.code(question['content'], language=None)
                st.info("👆 请手动选择上方文本并复制 (Ctrl+C)")
            
            st.markdown("---")
            
            # 选择模型并填写答案
            st.markdown("#### ✍️ 填写模型答案")
            
            if selected_model != "全部":
                target_models = [m for m in models_data["models"] if m["name"] == selected_model]
            else:
                target_models = models_data["models"]
            
            for model in target_models:
                existing_answer = get_answer(question['id'], model['id'])
                
                with st.container():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{model['icon']} {model['name']}** ([前往官网]({model['url']}))")
                    with col_b:
                        if existing_answer and existing_answer.get('score') is not None:
                            score = existing_answer['score']
                            st.markdown(f"<span class='score-badge score-{score}'>{score}分</span>", unsafe_allow_html=True)
                    
                    default_value = existing_answer['answer'] if existing_answer else ""
                    answer_text = st.text_area(
                        f"回答内容",
                        value=default_value,
                        height=150,
                        key=f"answer_{question['id']}_{model['id']}",
                        placeholder=f"将 {model['name']} 的回答粘贴到这里..."
                    )
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("💾 保存", key=f"save_{question['id']}_{model['id']}"):
                            if answer_text.strip():
                                save_answer(question['id'], model['id'], answer_text.strip())
                                st.success("✅ 已保存!")
                                st.rerun()
                            else:
                                st.warning("⚠️ 答案不能为空")
                    
                    with col2:
                        if existing_answer:
                            st.caption(f"📅 上次更新: {existing_answer['timestamp'][:19]}")
                    
                    st.markdown("---")


def render_scoring():
    """渲染评分打分页面"""
    st.markdown('<h1 class="main-title">⭐ 评分打分</h1>', unsafe_allow_html=True)
    
    questions_data = load_json(QUESTIONS_FILE)
    models_data = load_json(MODELS_FILE)
    rubric_data = load_json(SCORING_RUBRIC_FILE)
    
    if not questions_data or not models_data:
        st.error("无法加载数据文件")
        return
    
    # 筛选器
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_options = [m["name"] for m in models_data["models"]]
        selected_model_name = st.selectbox("🤖 选择模型", model_options)
        selected_model = next((m for m in models_data["models"] if m["name"] == selected_model_name), None)
    
    with col2:
        filter_options = ["全部", "未评分", "已评分"]
        filter_status = st.selectbox("📊 筛选状态", filter_options)
    
    with col3:
        question_ids = ["全部"] + [str(q["id"]) for q in questions_data["questions"]]
        selected_question = st.selectbox("📋 选择题号", question_ids)
    
    if not selected_model:
        return
    
    st.markdown("---")
    
    # 评分进度
    model_scores = get_model_scores(selected_model["id"])
    scored_count = len(model_scores)
    total_score = sum(model_scores.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已评分题目", f"{scored_count}/{TOTAL_QUESTIONS}")
    with col2:
        st.metric("当前总分", f"{total_score}/{MAX_TOTAL_SCORE}")
    with col3:
        avg_score = total_score / scored_count if scored_count > 0 else 0
        st.metric("平均分", f"{avg_score:.2f}")
    
    st.markdown("---")
    
    # 遍历题目进行评分
    for question in questions_data["questions"]:
        if selected_question != "全部" and str(question["id"]) != selected_question:
            continue
        
        answer = get_answer(question["id"], selected_model["id"])
        
        # 根据筛选状态过滤
        if filter_status == "未评分" and (answer and answer.get("score") is not None):
            continue
        if filter_status == "已评分" and (not answer or answer.get("score") is None):
            continue
        
        if not answer:
            continue
        
        dim_name = DIMENSION_NAMES.get(question['dimension'], question['dimension'])
        current_score = answer.get("score")
        
        with st.expander(
            f"**第 {question['id']} 题** - {question['title']} " + 
            (f"[{current_score}分]" if current_score is not None else "[未评分]"),
            expanded=(selected_question != "全部")
        ):
            # 显示题目
            st.markdown(f"**维度**: {dim_name}")
            
            with st.container():
                st.markdown("**📋 题目内容** (点击展开)")
                with st.expander("查看题目", expanded=False):
                    st.text(question['content'][:500] + "..." if len(question['content']) > 500 else question['content'])
            
            # 显示答案
            st.markdown("**📝 模型答案**")
            st.markdown(f"""
            <div class="answer-preview" style="max-height: 300px;">
                {answer['answer'][:2000]}{"..." if len(answer['answer']) > 2000 else ""}
            </div>
            """, unsafe_allow_html=True)
            
            # 显示已有评分详情（如果是自动评分的）
            if answer.get("scored_by") == "deepseek-auto":
                st.markdown("**🤖 AI自动评分详情**")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.success(f"✅ 优点: {answer.get('strengths', 'N/A')}")
                with col_s2:
                    st.warning(f"⚠️ 不足: {answer.get('weaknesses', 'N/A')}")
                if answer.get("comment"):
                    st.info(f"💬 评语: {answer.get('comment', '')}")
            
            # 显示评分标准
            if rubric_data:
                q_rubric = next((q for q in rubric_data["questions"] if q["id"] == question["id"]), None)
                if q_rubric:
                    with st.expander("📖 评分标准", expanded=False):
                        st.json(q_rubric.get("scoring_criteria", {}))
            
            # 评分输入
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                score = st.selectbox(
                    "评分 (0-5分)",
                    options=[None, 5, 4, 3, 2, 1, 0],
                    format_func=lambda x: "请选择" if x is None else f"{x}分",
                    index=0 if current_score is None else [None, 5, 4, 3, 2, 1, 0].index(current_score),
                    key=f"score_{question['id']}_{selected_model['id']}"
                )
            
            with col2:
                comment = st.text_input(
                    "评语 (可选)",
                    value=answer.get("comment", "") or "",
                    key=f"comment_{question['id']}_{selected_model['id']}"
                )
            
            if st.button("💾 保存评分", key=f"save_score_{question['id']}_{selected_model['id']}"):
                if score is not None:
                    save_score(question['id'], selected_model['id'], score, comment)
                    st.success(f"✅ 已保存评分: {score}分")
                    st.rerun()
                else:
                    st.warning("⚠️ 请选择评分")


def render_results():
    """渲染结果展示页面"""
    st.markdown('<h1 class="main-title">📊 结果展示</h1>', unsafe_allow_html=True)
    
    questions_data = load_json(QUESTIONS_FILE)
    models_data = load_json(MODELS_FILE)
    all_scores = get_all_scores()
    
    if not questions_data or not models_data:
        st.error("无法加载数据文件")
        return
    
    # 检查是否有评分数据
    has_scores = any(s["scored"] > 0 for s in all_scores.values())
    
    if not has_scores:
        st.warning("⚠️ 暂无评分数据，请先在「评分打分」页面进行评分")
        return
    
    # 选择展示方式
    view_type = st.selectbox(
        "📊 选择展示方式",
        ["🏆 总分排行榜", "📈 维度对比", "🎯 雷达图", "📋 详细得分表", "📉 得分分布", "🔍 题目横向对比"]
    )
    
    st.markdown("---")
    
    if view_type == "🏆 总分排行榜":
        render_leaderboard(all_scores, models_data)
    elif view_type == "📈 维度对比":
        render_dimension_comparison(all_scores, questions_data, models_data)
    elif view_type == "🎯 雷达图":
        render_radar_chart(all_scores, questions_data, models_data)
    elif view_type == "📋 详细得分表":
        render_score_table(all_scores, questions_data, models_data)
    elif view_type == "📉 得分分布":
        render_score_distribution(all_scores, questions_data, models_data)
    elif view_type == "🔍 题目横向对比":
        render_question_comparison(questions_data, models_data)


def render_leaderboard(all_scores, models_data):
    """渲染总分排行榜"""
    st.markdown("### 🏆 总分排行榜")
    
    # 准备排行数据
    leaderboard = []
    for model_id, data in all_scores.items():
        if data["scored"] > 0:
            leaderboard.append({
                "model_id": model_id,
                "name": data["name"],
                "icon": data["icon"],
                "total": data["total"],
                "scored": data["scored"],
                "max_possible": data["scored"] * SCORE_PER_QUESTION,
                "percentage": (data["total"] / (data["scored"] * SCORE_PER_QUESTION) * 100) if data["scored"] > 0 else 0
            })
    
    # 按总分排序
    leaderboard.sort(key=lambda x: x["total"], reverse=True)
    
    # 显示排行榜
    for rank, item in enumerate(leaderboard, 1):
        rank_class = f"rank-{rank}" if rank <= 3 else ""
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
        
        col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
        
        with col1:
            st.markdown(f"### {medal}")
        with col2:
            st.markdown(f"### {item['icon']} {item['name']}")
        with col3:
            st.markdown(f"### **{item['total']}** / {MAX_TOTAL_SCORE}")
        with col4:
            st.progress(item['percentage'] / 100)
            st.caption(f"{item['percentage']:.1f}% ({item['scored']}题已评)")
        
        st.markdown("---")
    
    # 柱状图
    if leaderboard:
        fig = px.bar(
            leaderboard,
            x="name",
            y="total",
            color="total",
            color_continuous_scale="Viridis",
            title="模型得分对比",
            labels={"name": "模型", "total": "总分"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_dimension_comparison(all_scores, questions_data, models_data):
    """渲染维度对比"""
    st.markdown("### 📈 维度对比")
    
    # 计算每个模型在每个维度的得分
    dimension_scores = {}
    
    for dim in questions_data["meta"]["dimensions"]:
        dim_id = dim["id"]
        dim_questions = dim["questions"]
        dimension_scores[dim_id] = {"name": dim["name"], "questions": dim_questions, "scores": {}}
        
        for model_id, data in all_scores.items():
            dim_total = 0
            dim_count = 0
            for q_id in dim_questions:
                if q_id in data["scores"]:
                    dim_total += data["scores"][q_id]
                    dim_count += 1
            
            if dim_count > 0:
                dimension_scores[dim_id]["scores"][model_id] = {
                    "total": dim_total,
                    "count": dim_count,
                    "max": dim_count * SCORE_PER_QUESTION,
                    "percentage": dim_total / (dim_count * SCORE_PER_QUESTION) * 100
                }
    
    # 创建数据框
    data_rows = []
    for dim_id, dim_data in dimension_scores.items():
        for model_id, score_data in dim_data["scores"].items():
            model_name = all_scores[model_id]["name"]
            data_rows.append({
                "维度": dim_data["name"],
                "模型": model_name,
                "得分": score_data["total"],
                "满分": score_data["max"],
                "得分率": score_data["percentage"]
            })
    
    if data_rows:
        df = pd.DataFrame(data_rows)
        
        # 分组柱状图
        fig = px.bar(
            df,
            x="维度",
            y="得分率",
            color="模型",
            barmode="group",
            title="各维度得分率对比",
            labels={"得分率": "得分率 (%)"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 热力图
        pivot_df = df.pivot(index="模型", columns="维度", values="得分率")
        fig_heat = px.imshow(
            pivot_df,
            color_continuous_scale="RdYlGn",
            title="得分率热力图",
            labels={"color": "得分率 (%)"}
        )
        fig_heat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9'
        )
        st.plotly_chart(fig_heat, use_container_width=True)


def render_radar_chart(all_scores, questions_data, models_data):
    """渲染雷达图"""
    st.markdown("### 🎯 能力雷达图")
    
    # 选择要对比的模型
    available_models = [m["name"] for m_id, m in all_scores.items() if m["scored"] > 0 for m in [all_scores[m_id]]]
    available_models = list(set(available_models))
    
    selected_models = st.multiselect(
        "选择要对比的模型",
        available_models,
        default=available_models[:3] if len(available_models) >= 3 else available_models
    )
    
    if not selected_models:
        st.warning("请选择至少一个模型")
        return
    
    # 计算每个维度的得分
    categories = [dim["name"] for dim in questions_data["meta"]["dimensions"]]
    
    fig = go.Figure()
    
    for model_name in selected_models:
        model_id = next((m_id for m_id, m in all_scores.items() if m["name"] == model_name), None)
        if not model_id:
            continue
        
        values = []
        for dim in questions_data["meta"]["dimensions"]:
            dim_total = 0
            dim_count = 0
            for q_id in dim["questions"]:
                if q_id in all_scores[model_id]["scores"]:
                    dim_total += all_scores[model_id]["scores"][q_id]
                    dim_count += 1
            
            if dim_count > 0:
                values.append(dim_total / (dim_count * SCORE_PER_QUESTION) * 100)
            else:
                values.append(0)
        
        # 闭合雷达图
        values.append(values[0])
        cats = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="模型能力雷达图",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f1f5f9'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_score_table(all_scores, questions_data, models_data):
    """渲染详细得分表"""
    st.markdown("### 📋 详细得分表")
    
    # 构建表格数据
    headers = ["题号", "维度"] + [m["name"] for m in models_data["models"] if m["id"] in all_scores]
    
    rows = []
    for question in questions_data["questions"]:
        row = [
            f"Q{question['id']}",
            DIMENSION_NAMES.get(question['dimension'], question['dimension'])
        ]
        
        for model in models_data["models"]:
            if model["id"] in all_scores:
                score = all_scores[model["id"]]["scores"].get(question["id"])
                row.append(score if score is not None else "-")
        
        rows.append(row)
    
    # 添加总分行
    total_row = ["总分", "-"]
    for model in models_data["models"]:
        if model["id"] in all_scores:
            total_row.append(all_scores[model["id"]]["total"])
    rows.append(total_row)
    
    df = pd.DataFrame(rows, columns=headers)
    
    # 样式化显示
    def color_score(val):
        if val == "-":
            return "background-color: #475569"
        if isinstance(val, (int, float)):
            if val >= 4:
                return "background-color: #22c55e; color: white"
            elif val >= 3:
                return "background-color: #eab308; color: black"
            elif val >= 2:
                return "background-color: #f97316; color: white"
            else:
                return "background-color: #ef4444; color: white"
        return ""
    
    st.dataframe(df, use_container_width=True, height=600)


def render_score_distribution(all_scores, questions_data, models_data):
    """渲染得分分布"""
    st.markdown("### 📉 得分分布")
    
    # 收集所有评分
    all_individual_scores = []
    for model_id, data in all_scores.items():
        for q_id, score in data["scores"].items():
            all_individual_scores.append({
                "模型": data["name"],
                "题号": f"Q{q_id}",
                "得分": score
            })
    
    if not all_individual_scores:
        st.warning("暂无评分数据")
        return
    
    df = pd.DataFrame(all_individual_scores)
    
    # 得分分布直方图
    fig = px.histogram(
        df,
        x="得分",
        color="模型",
        barmode="overlay",
        title="得分分布",
        labels={"得分": "得分", "count": "数量"}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f1f5f9'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 箱线图
    fig_box = px.box(
        df,
        x="模型",
        y="得分",
        color="模型",
        title="得分箱线图"
    )
    fig_box.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f1f5f9'
    )
    st.plotly_chart(fig_box, use_container_width=True)


def render_question_comparison(questions_data, models_data):
    """渲染题目横向对比"""
    st.markdown("### 🔍 题目横向对比")
    st.markdown("比较各模型在同一道题上的表现")
    
    # 选择题目
    question_options = [f"Q{q['id']}: {q['title']}" for q in questions_data["questions"]]
    selected = st.selectbox("选择题目", question_options)
    
    if not selected:
        return
    
    q_id = int(selected.split(":")[0].replace("Q", ""))
    question = next((q for q in questions_data["questions"] if q["id"] == q_id), None)
    
    if not question:
        return
    
    # 显示题目信息
    st.markdown(f"**维度**: {DIMENSION_NAMES.get(question['dimension'], question['dimension'])}")
    st.markdown(f"**难度**: {question.get('difficulty', 'N/A')}")
    
    with st.expander("📋 查看题目内容", expanded=False):
        st.text(question['content'])
    
    st.markdown("---")
    
    # 获取所有模型对该题的回答和评分
    answers_data = load_answers()
    model_answers = []
    
    for model in models_data["models"]:
        ans = get_answer(q_id, model["id"])
        if ans:
            model_answers.append({
                "model_id": model["id"],
                "model_name": model["name"],
                "model_icon": model["icon"],
                "answer": ans["answer"],
                "score": ans.get("score"),
                "comment": ans.get("comment", ""),
                "strengths": ans.get("strengths", ""),
                "weaknesses": ans.get("weaknesses", ""),
                "scored_by": ans.get("scored_by", "")
            })
    
    if not model_answers:
        st.warning("暂无该题的回答")
        return
    
    # 按得分排序
    model_answers.sort(key=lambda x: x["score"] if x["score"] is not None else -1, reverse=True)
    
    # 显示排名
    st.markdown("#### 📊 得分排名")
    
    for rank, ma in enumerate(model_answers, 1):
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank}"))
        score_display = f"{ma['score']}/5" if ma['score'] is not None else "未评分"
        
        with st.expander(f"{medal} {ma['model_icon']} {ma['model_name']} - **{score_display}**", expanded=(rank <= 3)):
            # 评分详情
            if ma['score'] is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    score_color = ["#ef4444", "#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"][ma['score']]
                    st.markdown(f"""
                    <div style="text-align:center; padding:1rem; background:{score_color}; border-radius:8px;">
                        <div style="font-size:2rem; font-weight:bold; color:white;">{ma['score']}</div>
                        <div style="color:white;">得分</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if ma.get('strengths'):
                        st.success(f"✅ {ma['strengths']}")
                with col3:
                    if ma.get('weaknesses'):
                        st.warning(f"⚠️ {ma['weaknesses']}")
                
                if ma.get('comment'):
                    st.info(f"💬 {ma['comment']}")
                
                if ma.get('scored_by') == 'deepseek-auto':
                    st.caption("🤖 由 DeepSeek 自动评分")
            
            # 显示回答
            st.markdown("**回答内容:**")
            st.markdown(f"""
            <div class="answer-preview" style="max-height:300px; overflow-y:auto;">
                {ma['answer'][:3000]}{"..." if len(ma['answer']) > 3000 else ""}
            </div>
            """, unsafe_allow_html=True)
    
    # 对比图表
    if any(ma['score'] is not None for ma in model_answers):
        st.markdown("---")
        st.markdown("#### 📈 得分对比图")
        
        chart_data = [
            {"模型": ma['model_name'], "得分": ma['score'] or 0}
            for ma in model_answers
        ]
        
        fig = px.bar(
            chart_data,
            x="模型",
            y="得分",
            color="得分",
            color_continuous_scale="RdYlGn",
            range_color=[0, 5],
            title=f"Q{q_id} 各模型得分对比"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9',
            yaxis_range=[0, 5.5]
        )
        st.plotly_chart(fig, use_container_width=True)


def render_data_view():
    """渲染数据查看页面"""
    st.markdown('<h1 class="main-title">📋 数据查看</h1>', unsafe_allow_html=True)
    
    questions_data = load_json(QUESTIONS_FILE)
    models_data = load_json(MODELS_FILE)
    answers_data = load_answers()
    
    if not questions_data or not models_data:
        st.error("无法加载数据文件")
        return
    
    # 数据概览
    st.markdown("### 📈 数据概览")
    
    # 构建完成度矩阵
    completion_matrix = {}
    for model in models_data["models"]:
        completion_matrix[model["id"]] = {}
        for question in questions_data["questions"]:
            answer = get_answer(question["id"], model["id"])
            if answer:
                score = answer.get("score")
                if score is not None:
                    completion_matrix[model["id"]][question["id"]] = f"✅{score}"
                else:
                    completion_matrix[model["id"]][question["id"]] = "📝"
            else:
                completion_matrix[model["id"]][question["id"]] = "❌"
    
    # 显示矩阵
    st.markdown("#### 📋 答案与评分状态矩阵")
    st.caption("✅已评分 | 📝有答案未评分 | ❌无答案")
    
    # 构建表格HTML
    header_cells = "<th>模型</th>" + "".join([f"<th>Q{q['id']}</th>" for q in questions_data["questions"]])
    
    rows = []
    for model in models_data["models"]:
        row_cells = f"<td><strong>{model['icon']} {model['name']}</strong></td>"
        for question in questions_data["questions"]:
            status = completion_matrix[model["id"]][question["id"]]
            row_cells += f"<td style='text-align: center;'>{status}</td>"
        rows.append(f"<tr>{row_cells}</tr>")
    
    table_html = f"""
    <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px;">
            <thead style="background: #334155;">
                <tr>{header_cells}</tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 按题目查看答案对比
    st.markdown("### 🔍 答案对比")
    
    question_options = [f"Q{q['id']}: {q['title']}" for q in questions_data["questions"]]
    selected = st.selectbox("选择题目", question_options)
    
    if selected:
        q_id = int(selected.split(":")[0].replace("Q", ""))
        question = next((q for q in questions_data["questions"] if q["id"] == q_id), None)
        
        if question:
            st.markdown(f"**题目内容**: {question['content'][:200]}..." if len(question['content']) > 200 else f"**题目内容**: {question['content']}")
            
            st.markdown("---")
            
            cols = st.columns(2)
            for idx, model in enumerate(models_data["models"]):
                with cols[idx % 2]:
                    answer = get_answer(q_id, model["id"])
                    score_badge = ""
                    if answer and answer.get("score") is not None:
                        score = answer["score"]
                        score_badge = f" <span class='score-badge score-{score}'>{score}分</span>"
                    
                    st.markdown(f"#### {model['icon']} {model['name']}{score_badge}", unsafe_allow_html=True)
                    if answer:
                        st.markdown(f"""
                        <div class="answer-preview">
                            {answer['answer'][:500]}{"..." if len(answer['answer']) > 500 else ""}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"📅 {answer['timestamp'][:19]}")
                    else:
                        st.info("暂无答案")
    
    st.markdown("---")
    
    # 导出功能
    st.markdown("### 📤 数据导出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 导出所有答案 (JSON)"):
            st.download_button(
                label="下载 answers.json",
                data=json.dumps(answers_data, ensure_ascii=False, indent=2),
                file_name="answers_export.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📥 导出评分结果 (CSV)"):
            # 准备CSV数据
            csv_rows = []
            for ans in answers_data["answers"]:
                csv_rows.append({
                    "question_id": ans["question_id"],
                    "model_id": ans["model_id"],
                    "score": ans.get("score", ""),
                    "comment": ans.get("comment", ""),
                    "timestamp": ans.get("timestamp", "")
                })
            
            if csv_rows:
                df = pd.DataFrame(csv_rows)
                st.download_button(
                    label="下载 scores.csv",
                    data=df.to_csv(index=False),
                    file_name="scores_export.csv",
                    mime="text/csv"
                )


def render_settings():
    """渲染设置页面"""
    st.markdown('<h1 class="main-title">⚙️ 设置</h1>', unsafe_allow_html=True)
    
    st.markdown("### 📁 数据管理")
    
    # 显示文件状态
    files = [
        ("questions.json", QUESTIONS_FILE),
        ("models.json", MODELS_FILE),
        ("answers.json", ANSWERS_FILE),
        ("scoring_rubric.json", SCORING_RUBRIC_FILE)
    ]
    
    for name, path in files:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "✅ 存在" if exists else "❌ 不存在"
        st.markdown(f"- **{name}**: {status} ({size/1024:.1f} KB)")
    
    st.markdown("---")
    
    st.markdown("### 🔄 重置数据")
    
    st.warning("⚠️ 以下操作不可逆，请谨慎操作！")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ 清空所有答案", type="secondary"):
            confirm = st.checkbox("确认清空所有答案数据", key="confirm_clear_answers")
            if confirm:
                empty_data = {
                    "meta": {
                        "title": "模型回答记录",
                        "version": "1.0",
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat()
                    },
                    "answers": []
                }
                save_json(ANSWERS_FILE, empty_data)
                st.success("✅ 已清空所有答案!")
                st.rerun()
    
    with col2:
        if st.button("🗑️ 清空所有评分", type="secondary"):
            confirm = st.checkbox("确认清空所有评分数据", key="confirm_clear_scores")
            if confirm:
                answers_data = load_answers()
                for ans in answers_data["answers"]:
                    ans["score"] = None
                    ans["comment"] = None
                    if "scored_at" in ans:
                        del ans["scored_at"]
                save_json(ANSWERS_FILE, answers_data)
                st.success("✅ 已清空所有评分!")
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ℹ️ 关于")
    st.markdown(f"""
    **LLM 测评平台** v2.0
    
    用于测试和比较不同大语言模型在各维度上的表现。
    
    - 📝 {TOTAL_QUESTIONS}道测评题目
    - 🤖 多个主流大语言模型
    - ⭐ 每题{SCORE_PER_QUESTION}分，总分{MAX_TOTAL_SCORE}分
    - 📊 多维度可视化分析
    - 💾 JSON数据持久化
    """)


def main():
    """主函数"""
    page = render_sidebar()
    
    if page == "🏠 首页":
        render_home()
    elif page == "📝 题目测评":
        render_questions()
    elif page == "⭐ 评分打分":
        render_scoring()
    elif page == "📊 结果展示":
        render_results()
    elif page == "📋 数据查看":
        render_data_view()
    elif page == "⚙️ 设置":
        render_settings()


if __name__ == "__main__":
    main()
