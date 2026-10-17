#!/usr/bin/env python3
"""
Publication-Grade PDF Documentation Generator for Delto-3F In-Hand Manipulation.
Converts learning.md into a beautifully styled, KaTeX-rendered, paged PDF report.
"""

import os
import re
import subprocess
import sys
import markdown
from pygments.formatters import HtmlFormatter

def generate_pdf():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    learning_path = os.path.join(project_root, "learning.md")
    output_pdf = os.path.join(project_root, "IN_HAND_MANIPULATION_DOCUMENTATION.pdf")
    temp_html = os.path.join("/tmp", "learning_documentation.html")

    print(f"[PDF Generator] Reading source: {learning_path}")
    with open(learning_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Protect LaTeX math from markdown mangling
    math_placeholders = {}
    math_idx = [0]

    def save_display_math(match):
        key = f"XXLATEXDISPLAY{math_idx[0]}XX"
        math_idx[0] += 1
        math_placeholders[key] = match.group(0)
        return f"\n\n{key}\n\n"

    def save_inline_math(match):
        key = f"XXLATEXINLINE{math_idx[0]}XX"
        math_idx[0] += 1
        math_placeholders[key] = match.group(0)
        return key

    # Protect Mermaid code blocks
    mermaid_blocks = {}
    mermaid_idx = [0]
    def save_mermaid(match):
        key = f"XXMERMAIDBLOCK{mermaid_idx[0]}XX"
        mermaid_idx[0] += 1
        diagram_code = match.group(1).strip()
        mermaid_blocks[key] = f'<div class="mermaid-container"><pre class="mermaid">\n{diagram_code}\n</pre></div>'
        return f"\n\n{key}\n\n"

    # Protect Mermaid blocks
    content = re.sub(r'```mermaid\s*\n(.*?)\n```', save_mermaid, content, flags=re.DOTALL)

    # Protect display math ($$...$$)
    content = re.sub(r'\$\$(.*?)\$\$', save_display_math, content, flags=re.DOTALL)

    # Protect inline math ($...$) - negative lookbehind/lookahead for escaped dollar
    content = re.sub(r'(?<!\\)\$(.+?)(?<!\\)\$', save_inline_math, content)

    # 2. Convert markdown to HTML with Pygments code highlighting
    md = markdown.Markdown(
        extensions=['tables', 'fenced_code', 'codehilite', 'toc'],
        extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'guess_lang': False,
                'pygments_style': 'one-dark',
            }
        }
    )
    pygments_css = HtmlFormatter(style='one-dark').get_style_defs('.highlight')
    body_html = md.convert(content)

    # 3. Restore Mermaid and LaTeX math
    for key, val in mermaid_blocks.items():
        body_html = body_html.replace(f"<p>{key}</p>", val).replace(key, val)

    for key, val in math_placeholders.items():
        # Clean up any surrounding paragraph tags if it was display math
        body_html = body_html.replace(f"<p>{key}</p>", f"<div>{val}</div>").replace(key, val)

    # 4. Assemble Complete Professional Document
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Delto-3F Autonomous In-Hand Manipulation — Technical Learning & Reward Engineering Report</title>

<!-- KaTeX for high-performance math rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

<!-- Mermaid.js for architecture diagrams -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>

<style>
    @page {
        size: A4 portrait;
        margin: 20mm 16mm 20mm 16mm;
        @top-center {
            content: "Delto-3F In-Hand Manipulation — Systematic Learning & Reward Engineering";
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 8pt;
            color: #64748b;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
            width: 100%;
        }
        @bottom-right {
            content: "Page " counter(page);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 8pt;
            font-weight: 600;
            color: #475569;
        }
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1e293b;
        background-color: #ffffff;
        font-size: 9.5pt;
        line-height: 1.55;
        margin: 0;
        padding: 0;
    }

    /* Cover Page */
    .cover-page {
        page-break-after: always;
        height: 92vh;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding-top: 40px;
        box-sizing: border-box;
    }

    .cover-badge {
        display: inline-block;
        padding: 6px 14px;
        background: #0284c7;
        color: #ffffff;
        border-radius: 20px;
        font-size: 9pt;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 24px;
    }

    .cover-title {
        font-size: 28pt;
        font-weight: 800;
        line-height: 1.15;
        color: #0f172a;
        margin: 0 0 16px 0;
    }

    .cover-subtitle {
        font-size: 13.5pt;
        line-height: 1.4;
        color: #475569;
        margin: 0 0 32px 0;
        font-weight: 400;
    }

    .specs-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0284c7;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 30px;
    }

    .specs-card h3 {
        margin: 0 0 10px 0;
        font-size: 11pt;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .specs-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px 16px;
        font-size: 9pt;
    }

    .specs-grid-item strong {
        color: #0f172a;
    }

    .cover-footer {
        border-top: 1px solid #e2e8f0;
        padding-top: 16px;
        font-size: 8.5pt;
        color: #64748b;
        display: flex;
        justify-content: space-between;
    }

    /* Typography & Section Dividers */
    h1 {
        font-size: 20pt;
        font-weight: 800;
        color: #0f172a;
        margin-top: 28px;
        margin-bottom: 12px;
        border-bottom: 2px solid #0284c7;
        padding-bottom: 6px;
        page-break-after: avoid;
    }

    h2 {
        font-size: 13.5pt;
        font-weight: 700;
        color: #0f172a;
        margin-top: 24px;
        margin-bottom: 10px;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 4px;
        page-break-after: avoid;
    }

    h3 {
        font-size: 11pt;
        font-weight: 600;
        color: #1e293b;
        margin-top: 16px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }

    h4 {
        font-size: 10pt;
        font-weight: 600;
        color: #334155;
        margin-top: 12px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }

    p {
        margin: 0 0 10px 0;
        text-align: justify;
    }

    ul, ol {
        margin: 0 0 10px 0;
        padding-left: 20px;
    }

    li {
        margin-bottom: 4px;
    }

    /* High-Aesthetic Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0 18px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        overflow: hidden;
    }

    th {
        background-color: #0f172a;
        color: #ffffff;
        font-weight: 600;
        text-align: left;
        padding: 7px 10px;
        border: 1px solid #334155;
    }

    td {
        padding: 6px 10px;
        border: 1px solid #e2e8f0;
        vertical-align: middle;
    }

    tr:nth-child(even) {
        background-color: #f8fafc;
    }

    tr:hover {
        background-color: #f1f5f9;
    }

    /* Code Blocks & Pygments Syntax Highlighting */
    {pygments_css}

    .highlight {
        background-color: #0f172a !important;
        border-radius: 6px;
        margin: 10px 0 14px 0;
        border: 1px solid #1e293b;
        overflow-x: auto;
        page-break-inside: avoid;
    }

    .highlight pre, pre {
        background-color: #0f172a !important;
        color: #abb2bf;
        padding: 10px 14px !important;
        border-radius: 6px;
        font-family: "JetBrains Mono", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 8pt;
        line-height: 1.45;
        overflow-x: auto;
        page-break-inside: avoid;
        border: none !important;
        margin: 0 !important;
    }

    .highlight code, pre code {
        background-color: transparent !important;
        color: inherit !important;
        padding: 0 !important;
        border: none !important;
        font-family: inherit;
        font-size: inherit;
    }

    code {
        font-family: "JetBrains Mono", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 8.5pt;
        background-color: #f1f5f9;
        color: #0369a1;
        padding: 1px 4px;
        border-radius: 3px;
        border: 1px solid #e2e8f0;
    }

    /* Blockquotes / Callouts */
    blockquote {
        border-left: 4px solid #0284c7;
        margin: 12px 0;
        padding: 8px 14px;
        background-color: #f0f9ff;
        color: #0369a1;
        border-radius: 0 4px 4px 0;
        page-break-inside: avoid;
    }

    /* KaTeX Adjustments */
    .katex-display {
        margin: 10px 0 !important;
        padding: 6px 0;
        overflow-x: auto;
        overflow-y: hidden;
    }

    .katex {
        font-size: 1.02em !important;
    }

    /* Mermaid diagrams */
    .mermaid-container {
        text-align: center;
        margin: 16px 0;
        page-break-inside: avoid;
    }

    .mermaid {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Page Breaks */
    .page-break {
        page-break-after: always;
    }

    hr {
        border: none;
        height: 1px;
        background-color: #cbd5e1;
        margin: 20px 0;
    }
</style>
</head>
<body>

<!-- Cover Page -->
<div class="cover-page">
    <div>
        <div class="cover-badge">Research & Engineering Monograph</div>
        <div class="cover-title">Autonomous In-Hand Manipulation on the Delto-3F Gripper</div>
        <div class="cover-subtitle">
            Systematic Evolution of Deep Reinforcement Learning Policies, Sim-to-Real Domain Randomization, Biomechanical Safety Constraints, and Multi-Stage Reward Engineering
        </div>

        <div class="specs-card">
            <h3>Executive Architecture & Champion Verification</h3>
            <div class="specs-grid">
                <div class="specs-grid-item"><strong>Hardware Platform:</strong> Delto Gripper-3F (12 Actuated Degrees of Freedom)</div>
                <div class="specs-grid-item"><strong>RL Algorithm:</strong> Proximal Policy Optimization (PPO) + Huber Critic Loss</div>
                <div class="specs-grid-item"><strong>Action Representation:</strong> 12-DOF Target Position PD Servos ($K_p=50, K_d=2$)</div>
                <div class="specs-grid-item"><strong>Action Rate Smoothing:</strong> 1st-Order EMA Filter ($\alpha = 0.50$) at 50 Hz</div>
                <div class="specs-grid-item"><strong>Manipulated Geometries:</strong> Cube (45 mm), Cylinder (70 mm), Cuboid (70 mm)</div>
                <div class="specs-grid-item"><strong>Observation Space:</strong> 39-dim Normalized Vector (Non-Privileged)</div>
                <div class="specs-grid-item"><strong>Hardware Safety:</strong> Zero Palm Intrusions ($+12$ mm Floor), Anti-Clench ($163^\circ$)</div>
                <div class="specs-grid-item"><strong>Sim-to-Real Latency:</strong> CAN Bus ($0\text{--}20$ ms) + ArUco Vision ($20\text{--}40$ ms)</div>
                <div class="specs-grid-item"><strong>Silicone Contact Physics:</strong> Dynamic Sliding Friction $\mu \in [2.1, 3.9]$ ($\pm 30\%$)</div>
                <div class="specs-grid-item"><strong>Champion Checkpoint:</strong> <code>best_model_with_dr.pth</code> (alias: <code>best_model.pth</code>)</div>
                <div class="specs-grid-item"><strong>Clean Baseline:</strong> <code>best_model_clean.pth</code> (Ideal kinematics)</div>
                <div class="specs-grid-item"><strong>Champion Score:</strong> <code>0.7519</code> (Stage 15 Iteration 500 Peak)</div>
                <div class="specs-grid-item"><strong>Sim-to-Real Success:</strong> <code>70.0%</code> Grand Success | <code>86.7%</code> Range 1 Success</div>
            </div>
        </div>
    </div>

    <div class="cover-footer">
        <div><strong>Hardware Platform:</strong> Delto-3F Gripper (12-DOF)</div>
        <div><strong>Physics Simulation:</strong> MuJoCo 3.x Engine</div>
        <div><strong>Release Date:</strong> October 2026</div>
    </div>
</div>

<!-- Main Body -->
<div class="content-body">
{body_html}
</div>

<script>
    document.addEventListener("DOMContentLoaded", function() {
        // Render math via KaTeX
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false}
            ],
            throwOnError: false
        });

        // Initialize Mermaid
        mermaid.initialize({
            startOnLoad: true,
            theme: 'neutral',
            flowchart: { useMaxWidth: true, htmlLabels: true }
        });

        window.status = "ready";
    });
</script>
</body>
</html>
"""
    html_document = html_template.replace('{body_html}', body_html).replace('{pygments_css}', pygments_css)

    print(f"[PDF Generator] Writing intermediate HTML to: {temp_html}")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_document)

    # 5. Render to PDF via Headless Chrome
    print(f"[PDF Generator] Invoking Google Chrome headless to compile: {output_pdf}")
    chrome_cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=12000",
        f"--print-to-pdf={output_pdf}",
        temp_html
    ]

    res = subprocess.run(chrome_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[PDF Generator ERROR]: {res.stderr}")
        sys.exit(1)

    if os.path.exists(output_pdf):
        file_size_mb = os.path.getsize(output_pdf) / (1024 * 1024)
        print(f"[PDF Generator SUCCESS]: Created {output_pdf} ({file_size_mb:.2f} MB)")
    else:
        print("[PDF Generator ERROR]: Output PDF was not created.")
        sys.exit(1)

if __name__ == "__main__":
    generate_pdf()
