import ast
import datetime
import html
import json
import math
import os
import random
import re
from threading import Thread
from urllib import parse, request
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from knowledge.doc_learning import LocalDocKnowledgeBase

APP_NAME = "团子"
APP_NAME_EN = "Tuanzi"

st.set_page_config(page_title=APP_NAME, page_icon="🧠", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        :root {
            --bg-1: #130c0d;
            --bg-2: #261416;
            --panel: rgba(32, 17, 19, 0.84);
            --panel-strong: rgba(45, 21, 24, 0.92);
            --line: rgba(255, 192, 195, 0.16);
            --text: #f7eeeb;
            --muted: #d0b2ac;
            --accent: #f68b93;
            --accent-2: #ffb18e;
            --accent-3: #ffddd1;
            --good: #9ae1c0;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(247, 115, 125, 0.24), transparent 24%),
                radial-gradient(circle at 82% 8%, rgba(255, 177, 142, 0.15), transparent 22%),
                radial-gradient(circle at 78% 82%, rgba(158, 225, 192, 0.10), transparent 18%),
                linear-gradient(180deg, var(--bg-1) 0%, #190d0f 30%, var(--bg-2) 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(24, 12, 14, 0.98), rgba(42, 20, 24, 0.95)) !important;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        .stMainBlockContainer > div:first-child {
            margin-top: -32px !important;
        }

        .stApp > div:last-child {
            margin-bottom: -28px !important;
        }

        .stSlider [data-baseweb="slider"] > div > div {
            background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
        }

        .stCheckbox label, .stRadio label, .stSelectbox label {
            font-weight: 600 !important;
        }

        .stButton > button {
            border-radius: 999px !important;
            border: 1px solid rgba(255, 202, 197, 0.18) !important;
            background: rgba(255, 255, 255, 0.03) !important;
            color: var(--text) !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button:hover {
            border-color: rgba(255, 202, 197, 0.35) !important;
            background: rgba(255, 255, 255, 0.07) !important;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 24% 24%, rgba(246, 139, 147, 0.24), transparent 18%),
                radial-gradient(circle at 74% 22%, rgba(255, 177, 142, 0.18), transparent 16%),
                linear-gradient(135deg, rgba(58, 26, 31, 0.92), rgba(25, 13, 16, 0.96));
            border: 1px solid rgba(255, 202, 197, 0.12);
            border-radius: 28px;
            padding: 30px 28px 24px 28px;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
            margin: 10px 0 18px 0;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -80px -80px auto;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(246, 139, 147, 0.22), transparent 62%);
            filter: blur(6px);
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 110px 1fr;
            gap: 18px;
            align-items: center;
        }

        .hero-orb {
            width: 110px;
            height: 110px;
            border-radius: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 44px;
            font-weight: 800;
            letter-spacing: 2px;
            color: white;
            background:
                radial-gradient(circle at 32% 28%, rgba(255,255,255,0.35), transparent 18%),
                linear-gradient(135deg, #ff8d95, #ffb08a 70%, #ffd9cc);
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.32),
                0 16px 40px rgba(246, 139, 147, 0.26);
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--accent-3);
            opacity: 0.88;
            margin-bottom: 10px;
        }

        .hero-title {
            font-size: 34px;
            line-height: 1.05;
            margin: 0;
            font-weight: 900;
            color: var(--text);
        }

        .hero-subtitle {
            font-size: 15px;
            line-height: 1.75;
            color: var(--muted);
            margin: 12px 0 0 0;
            max-width: 780px;
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 16px;
        }

        .hero-chip {
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 202, 197, 0.12);
            color: var(--accent-3);
            font-size: 12px;
        }

        .section-title {
            margin: 24px 0 10px 0;
            font-size: 12px;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #f3c3b3;
            opacity: 0.82;
        }

        .layer-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 0 0 12px 0;
        }

        .layer-card {
            background: linear-gradient(180deg, rgba(48, 23, 26, 0.84), rgba(27, 14, 16, 0.88));
            border: 1px solid rgba(255, 202, 197, 0.10);
            border-radius: 20px;
            padding: 16px 14px;
            min-height: 140px;
        }

        .layer-index {
            color: var(--accent);
            font-size: 11px;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .layer-name {
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 10px;
            color: var(--text);
        }

        .layer-body {
            color: var(--muted);
            line-height: 1.7;
            font-size: 14px;
        }

        .journal-shell {
            background: linear-gradient(180deg, rgba(26, 14, 16, 0.80), rgba(21, 11, 13, 0.92));
            border: 1px solid rgba(255, 202, 197, 0.09);
            border-radius: 22px;
            padding: 20px 18px 4px 18px;
            margin: 10px 0 18px 0;
        }

        .journal-item {
            padding: 0 0 14px 0;
            margin: 0 0 14px 0;
            border-bottom: 1px solid rgba(255, 202, 197, 0.08);
        }

        .journal-item:last-child {
            border-bottom: 0;
            margin-bottom: 0;
        }

        .journal-item h4 {
            margin: 0 0 8px 0;
            font-size: 18px;
            color: var(--accent-3);
        }

        .journal-item p {
            margin: 0;
            color: var(--muted);
            line-height: 1.8;
        }

        .reality-note {
            margin: 8px 0 0 0;
            color: #a7dac3;
            font-size: 13px;
        }

        .assistant-block {
            background: linear-gradient(180deg, rgba(34, 17, 19, 0.92), rgba(25, 13, 15, 0.95));
            border: 1px solid rgba(255, 202, 197, 0.10);
            border-radius: 24px;
            padding: 14px 16px;
            margin: 10px 0;
            color: var(--text);
        }

        .user-bubble {
            display: flex;
            justify-content: flex-end;
            margin: 10px 0;
        }

        .user-bubble > div {
            display: inline-block;
            max-width: min(720px, 100%);
            padding: 10px 14px;
            border-radius: 22px;
            background: linear-gradient(135deg, #ee7b84, #f9aa8c);
            color: #fff8f5;
            box-shadow: 0 10px 24px rgba(246, 139, 147, 0.18);
        }

        .tool-box {
            background: rgba(79, 109, 129, 0.22);
            border: 1px solid rgba(160, 190, 215, 0.24);
            padding: 10px 12px;
            border-radius: 14px;
            margin: 8px 0;
        }

        .tool-box.success {
            background: rgba(76, 120, 99, 0.22);
            border-color: rgba(150, 210, 175, 0.24);
        }

        details {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 202, 197, 0.08);
            border-radius: 14px;
            padding: 10px 12px;
            margin: 8px 0;
        }

        details summary {
            cursor: pointer;
            color: #f2c8be;
        }

        @media (max-width: 860px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero-orb {
                width: 84px;
                height: 84px;
                font-size: 34px;
                border-radius: 24px;
            }

            .hero-title {
                font-size: 28px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

LANG_TEXTS = {
    "zh": {
        "project_name": APP_NAME,
        "settings": "神经设定",
        "history_rounds": "历史对话轮次",
        "max_length": "最大生成长度",
        "temperature": "温度",
        "thinking": "显式思考",
        "tools": "真实工具",
        "language": "语言",
        "send": "给团子发送一条消息",
        "disclaimer": "模型回答与联网工具结果都应二次核验",
        "think_tip": "显式展示模型思考片段；多轮对话和工具混用时仍可能抖动",
        "tool_select": "最多选择 4 个真实工具",
        "model": "脑体型号",
        "welcome": "我是 {model}，一颗偏爱脑结构与工具推理的仿生助手。",
        "subtitle": "项目现以“团子”对外呈现。界面参考了你给出的仿生脑文章，把头皮、脑膜、边缘系统、皮层和脑机接口这些层次重新组织成可交互的界面语言。",
        "chip_1": "五层外壳",
        "chip_2": "边缘系统",
        "chip_3": "皮层推理",
        "chip_4": "真实工具返回",
        "layers_title": "从外到里的脑层剖面",
        "journal_title": "仿生参考摘记",
        "journal_intro": "这部分不是机械科普卡片，而是把你给出的长文压缩成界面底稿：真正的大脑难看、柔软、密度惊人，而且我们对它的理解仍远未完成。",
        "tool_reality": "已移除天气、汇率、翻译的假返回，改为实时请求。",
        "thinking_done": "已思考",
        "thinking_in_progress": "思考中...",
        "tool_calling": "工具调用",
        "tool_called": "工具结果",
        "system_prompt": "你是团子，一个仿生脑风格的 AI 助手。回答要完整、准确、自然；如果需要使用工具，优先依赖真实工具结果，不要编造。",
    },
    "en": {
        "project_name": APP_NAME_EN,
        "settings": "Neural Settings",
        "history_rounds": "History Rounds",
        "max_length": "Max Length",
        "temperature": "Temperature",
        "thinking": "Thinking",
        "tools": "Live Tools",
        "language": "Language",
        "send": "Send a message to Tuanzi",
        "disclaimer": "Model output and live tool results should still be verified",
        "think_tip": "Expose model reasoning snippets; may still wobble during multi-turn tool usage",
        "tool_select": "Select up to 4 live tools",
        "model": "Brain Body",
        "welcome": "I am {model}, a biomimetic assistant built around brain anatomy and tool reasoning.",
        "subtitle": "The project is now presented externally as “Tuanzi”. The interface draws from your brain article: scalp, meninges, limbic system, cortex, and BMI trade-offs are reorganized into the chat surface.",
        "chip_1": "Five shell layers",
        "chip_2": "Limbic circuitry",
        "chip_3": "Cortical reasoning",
        "chip_4": "Live tool outputs",
        "layers_title": "Anatomy, Outside In",
        "journal_title": "Biomimetic Notes",
        "journal_intro": "This compresses your long reference text into interface language: the real brain is messy, soft, information-dense, and still poorly understood.",
        "tool_reality": "Weather, FX, and translation now use live requests instead of placeholders.",
        "thinking_done": "Reasoned",
        "thinking_in_progress": "Reasoning...",
        "tool_calling": "Tool Call",
        "tool_called": "Tool Result",
        "system_prompt": "You are Tuanzi, a biomimetic AI assistant. Answer clearly and completely. When tools are needed, prefer live tool outputs over guesses.",
    },
}

ANATOMY_LAYERS = {
    "zh": [
        ("Layer 01", "头皮与颅骨", "表面看起来像边界，实际只是最外层。头骨不是终点，上面和下面都还有大量结构。"),
        ("Layer 02", "硬脑膜", "坚韧、防水、会疼。很多剧烈头痛并不是脑组织在痛，而是这层在报警。"),
        ("Layer 03", "蛛网膜", "纤维与脑脊液一起给大脑减震。它让大脑不是漂着，而是被悬挂和缓冲。"),
        ("Layer 04", "软脑膜", "贴着脑表面的细致薄层，血管顺着褶皱埋入其中，外观因此总显得潮湿又复杂。"),
        ("Layer 05", "皮层", "真正的大规模感知、语言、规划与个性，大多发生在这张被折叠起来的“餐巾纸”里。"),
    ],
    "en": [
        ("Layer 01", "Scalp & Skull", "What looks like the boundary is only the shell. The skull is not the end point; there is a lot above and below it."),
        ("Layer 02", "Dura Mater", "Tough, waterproof, and pain-sensitive. A lot of severe headaches are dural, not cortical."),
        ("Layer 03", "Arachnoid", "Fibers plus cerebrospinal fluid suspend and cushion the brain instead of letting it simply float."),
        ("Layer 04", "Pia Mater", "A delicate membrane fused to the surface; vessels dive through folds so the outside never looks clean."),
        ("Layer 05", "Cortex", "Most large-scale sensing, language, planning, and personality live in this folded napkin of tissue."),
    ],
}

BRAIN_JOURNAL = {
    "zh": [
        ("果冻而不是机器外壳", "把其它结构剥开后，真正的大脑更像三磅重、只耗 20 瓦的软布丁。它不是冷硬硬件，而是一团会塌陷、会改线、会自组织的活物。"),
        ("三套主要控制系统", "脑干和小脑负责活着、平衡和基本动作；边缘系统负责情绪、生存和冲动；皮层负责语言、规划、推理和你以为属于“自己”的那部分意识。"),
        ("放大一千倍后会看到灾难级复杂度", "一个立方毫米皮层里就有约 4 万个神经元、数千万个突触、同量级神经胶质细胞以及密布血管。脑机接口面对的不是干净电路板，而是一锅活着的电子意大利面。"),
        ("BMI 的核心矛盾", "规模、解析度、侵入性三者很难兼得。fMRI 覆盖广但慢且粗，EEG 无创但空间分辨率很低，真正接近神经元级别就会快速提高侵入性。"),
    ],
    "en": [
        ("A Jelly, Not a Chassis", "Once the shell is stripped away, the brain looks more like a three-pound pudding running on 20 watts than a rigid machine."),
        ("Three Major Control Systems", "Brain stem and cerebellum keep you alive and coordinated; the limbic system handles survival, emotion, and impulse; cortex handles language, planning, reasoning, and the part you call 'self'."),
        ("At 1000x Scale the Complexity Explodes", "A single cubic millimeter of cortex holds around 40,000 neurons, tens of millions of synapses, matching glia, and dense vasculature. BMI does not face a neat board; it faces living spaghetti."),
        ("The BMI Trade-off", "Scale, resolution, and invasiveness are still hard to maximize together. fMRI sees broadly but slowly; EEG is non-invasive but spatially coarse; neuron-level access quickly becomes invasive."),
    ],
}

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式 / Evaluate a math expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间 / Get current time", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数 / Generate a random number", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本长度 / Count text length", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位转换 / Convert units", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气 / Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率 / Get exchange rate", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本 / Translate text", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

TOOL_SHORT_NAMES = {
    "calculate_math": "数学",
    "get_current_time": "时间",
    "random_number": "随机",
    "text_length": "字数",
    "unit_converter": "单位",
    "get_current_weather": "天气",
    "get_exchange_rate": "汇率",
    "translate_text": "翻译",
}

ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}
ALLOWED_NAMES = {"pi": math.pi, "e": math.e}
UNIT_GROUPS = {
    "length": {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1.0,
        "km": 1000.0,
        "in": 0.0254,
        "inch": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344,
    },
    "mass": {
        "mg": 0.001,
        "g": 1.0,
        "kg": 1000.0,
        "oz": 28.349523125,
        "lb": 453.59237,
    },
    "volume": {
        "ml": 0.001,
        "l": 1.0,
        "liter": 1.0,
        "litre": 1.0,
        "gal": 3.785411784,
    },
    "time": {
        "s": 1.0,
        "sec": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "hr": 3600.0,
        "day": 86400.0,
    },
}


def get_text(key):
    lang = st.session_state.get("lang", "zh")
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS["zh"].get(key, key))


def escape_html(text):
    return html.escape(str(text)).replace("\n", "<br>")


def render_user_message(text):
    return f'<div class="user-bubble"><div>{escape_html(text)}</div></div>'


def render_brain_shell():
    lang = st.session_state.get("lang", "zh")
    cards = []
    for index, title, body in ANATOMY_LAYERS[lang]:
        cards.append(
            f"""
            <div class="layer-card">
                <div class="layer-index">{escape_html(index)}</div>
                <div class="layer-name">{escape_html(title)}</div>
                <div class="layer-body">{escape_html(body)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="section-title">{escape_html(get_text("layers_title"))}</div><div class="layer-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_brain_journal():
    lang = st.session_state.get("lang", "zh")
    entries = []
    for title, body in BRAIN_JOURNAL[lang]:
        entries.append(
            f"""
            <div class="journal-item">
                <h4>{escape_html(title)}</h4>
                <p>{escape_html(body)}</p>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="journal-shell">
            <div class="section-title">{escape_html(get_text("journal_title"))}</div>
            <p class="hero-subtitle" style="margin:0 0 16px 0;">{escape_html(get_text("journal_intro"))}</p>
            {''.join(entries)}
            <p class="reality-note">{escape_html(get_text("tool_reality"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fetch_json(url, headers=None):
    req = request.Request(url, headers={"User-Agent": f"{APP_NAME_EN}/1.0", **(headers or {})})
    with request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def detect_source_lang(text):
    return "zh-CN" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def safe_eval(expression):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.id]
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("unsupported operator")
        if isinstance(node, ast.UnaryOp):
            value = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.USub):
                return -value
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS:
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            return ALLOWED_FUNCS[node.func.id](*[_eval(arg) for arg in node.args])
        raise ValueError("unsafe expression")

    parsed = ast.parse(expression, mode="eval")
    return _eval(parsed)


def normalize_unit(unit):
    return str(unit).strip().lower().replace("°", "")


def convert_temperature(value, from_unit, to_unit):
    if from_unit == to_unit:
        return value
    if from_unit == "c":
        celsius = value
    elif from_unit == "f":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "k":
        celsius = value - 273.15
    else:
        raise ValueError(f"unsupported temperature unit: {from_unit}")

    if to_unit == "c":
        return celsius
    if to_unit == "f":
        return celsius * 9 / 5 + 32
    if to_unit == "k":
        return celsius + 273.15
    raise ValueError(f"unsupported temperature unit: {to_unit}")


def convert_units(value, from_unit, to_unit):
    from_unit = normalize_unit(from_unit)
    to_unit = normalize_unit(to_unit)
    if from_unit in {"c", "f", "k"} or to_unit in {"c", "f", "k"}:
        return convert_temperature(float(value), from_unit, to_unit)
    for group in UNIT_GROUPS.values():
        if from_unit in group and to_unit in group:
            base_value = float(value) * group[from_unit]
            return base_value / group[to_unit]
    raise ValueError(f"unsupported conversion: {from_unit} -> {to_unit}")


def execute_tool(tool_name, args):
    try:
        if tool_name == "calculate_math":
            expr = str(args.get("expression", "")).strip()
            if not expr:
                raise ValueError("expression is required")
            return {"result": safe_eval(expr)}

        if tool_name == "get_current_time":
            tz_name = args.get("timezone", "Asia/Shanghai") or "Asia/Shanghai"
            try:
                now = datetime.datetime.now(ZoneInfo(tz_name))
            except ZoneInfoNotFoundError:
                raise ValueError(f"unknown timezone: {tz_name}")
            return {"result": now.strftime("%Y-%m-%d %H:%M:%S %Z")}

        if tool_name == "random_number":
            min_value = int(args.get("min", 0))
            max_value = int(args.get("max", 100))
            if min_value > max_value:
                raise ValueError("min must be <= max")
            return {"result": random.randint(min_value, max_value)}

        if tool_name == "text_length":
            return {"result": len(str(args.get("text", "")))}

        if tool_name == "unit_converter":
            value = float(args.get("value"))
            from_unit = args.get("from_unit", "")
            to_unit = args.get("to_unit", "")
            result = convert_units(value, from_unit, to_unit)
            return {"result": f"{value:g} {from_unit} = {result:.6g} {to_unit}"}

        if tool_name == "get_current_weather":
            city = str(args.get("city", "")).strip()
            if not city:
                raise ValueError("city is required")
            payload = fetch_json(f"https://wttr.in/{parse.quote(city)}?format=j1")
            current = payload["current_condition"][0]
            area_info = payload.get("nearest_area", [{}])[0]
            area = area_info.get("areaName", [{"value": city}])[0]["value"]
            weather_desc = current.get("weatherDesc", [{"value": ""}])[0]["value"]
            return {
                "result": f"{area}: {weather_desc}, {current['temp_C']}°C, feels like {current['FeelsLikeC']}°C, humidity {current['humidity']}%"
            }

        if tool_name == "get_exchange_rate":
            from_currency = str(args.get("from_currency", "")).upper().strip()
            to_currency = str(args.get("to_currency", "")).upper().strip()
            if not from_currency or not to_currency:
                raise ValueError("from_currency and to_currency are required")
            payload = fetch_json(f"https://open.er-api.com/v6/latest/{parse.quote(from_currency)}")
            if payload.get("result") != "success":
                raise ValueError(payload.get("error-type", "exchange rate service error"))
            rate = payload["rates"].get(to_currency)
            if rate is None:
                raise ValueError(f"unsupported currency: {to_currency}")
            return {"result": f"1 {from_currency} = {rate:.6f} {to_currency}"}

        if tool_name == "translate_text":
            text = str(args.get("text", "")).strip()
            target_lang = str(args.get("target_lang", "")).strip()
            if not text or not target_lang:
                raise ValueError("text and target_lang are required")
            try:
                params = parse.urlencode({"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": text})
                payload = fetch_json(
                    f"https://translate.googleapis.com/translate_a/single?{params}",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                translated = "".join(part[0] for part in payload[0] if part and part[0])
                detected = payload[2] if len(payload) > 2 else "auto"
                return {"result": translated, "detected_source": detected}
            except Exception:
                source_lang = detect_source_lang(text)
                memory_params = parse.urlencode({"q": text, "langpair": f"{source_lang}|{target_lang}"})
                payload = fetch_json(
                    f"https://api.mymemory.translated.net/get?{memory_params}",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                translated = payload["responseData"]["translatedText"]
                return {"result": translated, "detected_source": source_lang}

        return {"error": f"unknown tool: {tool_name}"}
    except Exception as exc:
        return {"error": str(exc)}


def process_assistant_content(content, is_streaming=False):
    if "<tool_call>" in content:
        def format_tool_call(match):
            try:
                tc = json.loads(match.group(1))
                name = escape_html(tc.get("name", "unknown"))
                args = escape_html(json.dumps(tc.get("arguments", {}), ensure_ascii=False))
                return f'<div class="tool-box"><div style="font-size:12px;opacity:.75;margin-bottom:6px;">{escape_html(get_text("tool_calling"))}</div><div><b>{name}</b>: {args}</div></div>'
            except Exception:
                return match.group(0)

        content = re.sub(r"<tool_call>(.*?)</tool_call>", format_tool_call, content, flags=re.DOTALL)

    if is_streaming and st.session_state.get("enable_thinking", False) and "</think>" not in content and "<think>" not in content:
        match = re.search(r"(\n\n(?:我是|您好|你好|I am|Hello)[^\n]*)", content)
        if match and match.start(1) > 5:
            index = match.start(1)
            think_part = escape_html(content[:index].strip())
            answer_part = content[index:]
            return f'<details open><summary>{escape_html(get_text("thinking_done"))}</summary><div style="color:#d4bbb5; margin-top:8px;">{think_part}</div></details>{answer_part}'
        if len(content) > 5:
            return f'<details open><summary>{escape_html(get_text("thinking_in_progress"))}</summary><div style="color:#d4bbb5; margin-top:8px;">{escape_html(content.strip())}</div></details>'

    if "<think>" in content and "</think>" in content:
        def format_think(match):
            think_content = match.group(2).strip()
            if think_content:
                return f'<details open><summary>{escape_html(get_text("thinking_done"))}</summary><div style="color:#d4bbb5; margin-top:8px;">{escape_html(think_content)}</div></details>'
            return ""

        content = re.sub(r"(<think>)(.*?)(</think>)", format_think, content, flags=re.DOTALL)

    if "<think>" in content and "</think>" not in content:
        def format_think_in_progress(match):
            return f'<details open><summary>{escape_html(get_text("thinking_in_progress"))}</summary><div style="color:#d4bbb5; margin-top:8px;">{escape_html(match.group(1).strip())}</div></details>'

        content = re.sub(r"<think>(.*?)$", format_think_in_progress, content, flags=re.DOTALL)

    if "<think>" not in content and "</think>" in content:
        def format_think_no_start(match):
            think_content = match.group(1).strip()
            if think_content:
                return f'<details open><summary>{escape_html(get_text("thinking_done"))}</summary><div style="color:#d4bbb5; margin-top:8px;">{escape_html(think_content)}</div></details>'
            return ""

        content = re.sub(r"(.*?)</think>", format_think_no_start, content, flags=re.DOTALL)

    return f'<div class="assistant-block">{content}</div>'


@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.half().eval().to(device)
    return model, tokenizer


@st.cache_resource
def load_doc_kb(doc_path):
    if not doc_path or not os.path.exists(doc_path):
        return None
    kb = LocalDocKnowledgeBase.from_path(doc_path)
    return kb if kb.is_ready() else None


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


script_dir = os.path.dirname(os.path.abspath(__file__))
doc_dir = os.path.abspath(os.path.join(script_dir, "..", "docs"))
MODEL_PATHS = {}
for directory in sorted(os.listdir(script_dir), reverse=True):
    full_path = os.path.join(script_dir, directory)
    if os.path.isdir(full_path) and not directory.startswith(".") and not directory.startswith("_"):
        if any(
            file.endswith((".bin", ".safetensors", ".pt"))
            or os.path.exists(os.path.join(full_path, "model.safetensors.index.json"))
            for file in os.listdir(full_path)
            if os.path.isfile(os.path.join(full_path, file))
        ):
            MODEL_PATHS[directory] = [full_path, directory]
if not MODEL_PATHS:
    MODEL_PATHS = {"No models found": ["", APP_NAME]}


def main():
    if "lang" not in st.session_state:
        st.session_state.lang = "zh"

    selected_model = st.sidebar.selectbox(get_text("model"), list(MODEL_PATHS.keys()), index=0)
    model_path = MODEL_PATHS[selected_model][0]

    lang_options = {"中文": "zh", "English": "en"}
    current_lang = st.session_state.get("lang", "zh")
    lang_index = 0 if current_lang == "zh" else 1
    lang_label = st.sidebar.radio("Language / 语言", list(lang_options.keys()), index=lang_index, horizontal=True)
    if lang_options[lang_label] != current_lang:
        st.session_state.lang = lang_options[lang_label]
        st.rerun()

    st.sidebar.markdown('<hr style="margin: 12px 0 16px 0; border-color: rgba(255,202,197,0.10);">', unsafe_allow_html=True)
    st.sidebar.markdown(f"### {get_text('settings')}")
    st.session_state.history_chat_num = st.sidebar.slider(get_text("history_rounds"), 0, 8, 0, step=2)
    st.session_state.max_new_tokens = st.sidebar.slider(get_text("max_length"), 256, 8192, 4096, step=64)
    st.session_state.temperature = st.sidebar.slider(get_text("temperature"), 0.6, 1.2, 0.90, step=0.01)

    st.sidebar.markdown('<hr style="margin: 12px 0 16px 0; border-color: rgba(255,202,197,0.10);">', unsafe_allow_html=True)
    st.session_state.enable_thinking = st.sidebar.checkbox(get_text("thinking"), value=False, help=get_text("think_tip"))
    st.session_state.selected_tools = []
    with st.sidebar.expander(get_text("tools"), expanded=True):
        st.caption(get_text("tool_select"))
        selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))
        for tool in TOOLS:
            name = tool["function"]["name"]
            short_name = TOOL_SHORT_NAMES.get(name, name)
            checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)))
            if checked and len(st.session_state.selected_tools) < 4:
                st.session_state.selected_tools.append(name)

    slogan = get_text("welcome").format(model=MODEL_PATHS[selected_model][1])
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-grid">
                <div class="hero-orb">团</div>
                <div>
                    <div class="hero-kicker">BIOMIMETIC CORTEX INTERFACE</div>
                    <h1 class="hero-title">{escape_html(get_text("project_name"))}</h1>
                    <p class="hero-subtitle">{escape_html(slogan)}</p>
                    <p class="hero-subtitle">{escape_html(get_text("subtitle"))}</p>
                    <div class="hero-meta">
                        <span class="hero-chip">{escape_html(get_text("chip_1"))}</span>
                        <span class="hero-chip">{escape_html(get_text("chip_2"))}</span>
                        <span class="hero-chip">{escape_html(get_text("chip_3"))}</span>
                        <span class="hero-chip">{escape_html(get_text("chip_4"))}</span>
                    </div>
                    <p class="reality-note">{escape_html(get_text("disclaimer"))}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_brain_shell()
    render_brain_journal()

    if not model_path:
        st.error("No local model directory was found under ./scripts/")
        return

    model, tokenizer = load_model_tokenizer(model_path)
    doc_kb = load_doc_kb(doc_dir)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages
    for message in messages:
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(render_user_message(message["content"]), unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder=get_text("send"))
    if prompt:
        st.markdown(render_user_message(prompt), unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
        st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

        placeholder = st.empty()
        setup_seed(random.randint(0, 2**32 - 1))

        tools = [tool for tool in TOOLS if tool["function"]["name"] in st.session_state.get("selected_tools", [])] or None
        system_prompt = [{"role": "system", "content": get_text("system_prompt")}]
        if doc_kb:
            doc_prompt, _ = doc_kb.build_system_prompt(prompt, top_k=4, min_score=0.05, max_chars=2200)
            if doc_prompt:
                system_prompt.insert(0, {"role": "system", "content": doc_prompt})
        st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if st.session_state.get("enable_thinking", False):
            template_kwargs["open_thinking"] = True
        if tools:
            template_kwargs["tools"] = tools

        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "top_p": 0.85,
            "streamer": streamer,
        }

        Thread(target=model.generate, kwargs=generation_kwargs).start()
        answer = ""
        for new_text in streamer:
            answer += new_text
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)

        full_answer = answer
        for _ in range(16):
            tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", answer, re.DOTALL)
            if not tool_calls:
                break

            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            tool_results = []
            for tc_str in tool_calls:
                try:
                    tool_call = json.loads(tc_str.strip())
                    result = execute_tool(tool_call.get("name", ""), tool_call.get("arguments", {}))
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    tool_results.append(
                        f'<div class="tool-box success"><div style="font-size:12px;opacity:.75;margin-bottom:6px;">{escape_html(get_text("tool_called"))}</div><div><b>{escape_html(tool_call.get("name", ""))}</b>: {escape_html(json.dumps(result, ensure_ascii=False))}</div></div>'
                    )
                except Exception as exc:
                    tool_results.append(
                        f'<div class="tool-box success"><div style="font-size:12px;opacity:.75;margin-bottom:6px;">{escape_html(get_text("tool_called"))}</div><div><b>error</b>: {escape_html(str(exc))}</div></div>'
                    )

            full_answer += "\n" + "\n".join(tool_results) + "\n"
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)

            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_length"] = inputs.input_ids.shape[1] + st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            Thread(target=model.generate, kwargs=generation_kwargs).start()
            answer = ""
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer

        answer = full_answer
        messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
