import html
import os
from pathlib import Path

import gradio as gr

from agent.agent_runner import ClinicalAgentRunner
from core.state import AppState


def update_settings_action(ncbi_key: str, hf_token: str, state: AppState) -> str:
    """Action callback: Updates runtime API keys and environment variables, persisting to .env."""
    updates: dict[str, str] = {}
    if ncbi_key and ncbi_key.strip():
        clean_ncbi = ncbi_key.strip()
        os.environ["NCBI_API_KEY"] = clean_ncbi
        updates["NCBI_API_KEY"] = clean_ncbi

    if hf_token and hf_token.strip():
        clean_token = hf_token.strip()
        os.environ["HF_TOKEN"] = clean_token
        state.hf_token = clean_token
        updates["HF_TOKEN"] = clean_token

    if updates:
        try:
            env_path = Path(".env")
            existing_lines: list[str] = []
            if env_path.exists():
                existing_lines = env_path.read_text(encoding="utf-8").splitlines()

            lines = [
                line
                for line in existing_lines
                if not any(line.startswith(f"{k}=") for k in updates)
            ]
            for k, v in updates.items():
                lines.append(f"{k}={v}")

            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            return f"""
    <div style='background:#fef2f2;border:1px solid #fca5a5;color:#991b1b;padding:12px;border-radius:8px;'>
        ❌ Failed to persist settings: {html.escape(str(e))}
    </div>
    """

    return """
    <div style='background:#f0fdf4;border:1px solid #86efac;color:#166534;padding:12px;border-radius:8px;'>
        ✅ Settings updated successfully. API credentials saved and ready for downstream PubMed and Hugging Face inference calls.
    </div>
    """


def test_hf_connection_action(hf_token: str, state: AppState) -> str:
    """Action callback: Tests live connectivity to Hugging Face Inference API."""
    token_to_test = (
        hf_token.strip() if hf_token else (state.hf_token or os.getenv("HF_TOKEN"))
    )
    success, msg, details = ClinicalAgentRunner.test_hf_connection(token=token_to_test)
    if success:
        return f"""
        <div style='background:#f0fdf4;border:1px solid #86efac;color:#166534;padding:14px;border-radius:8px;'>
            <strong>{html.escape(msg)}</strong>
            <div style='margin-top:6px;font-size:0.85rem;color:#15803d;'>
                Model: <code>{html.escape(details.get("model", ""))}</code> | Latency: <code>{details.get("elapsed_ms", 0)}ms</code>
            </div>
        </div>
        """
    else:
        return f"""
        <div style='background:#fef2f2;border:1px solid #fca5a5;color:#991b1b;padding:14px;border-radius:8px;'>
            <strong>{html.escape(msg)}</strong>
            <div style='margin-top:6px;font-size:0.85rem;color:#b91c1c;'>
                Please verify that your Hugging Face Token has <code>Read</code> permissions at <a href='https://huggingface.co/settings/tokens' target='_blank' style='text-decoration:underline;'>huggingface.co/settings/tokens</a>.
            </div>
        </div>
        """


def create_settings_view(app_state: gr.State) -> tuple[gr.Tab, dict[str, gr.Component]]:
    """
    Constructs the Gradio Native Tab for Settings and System Diagnostics.
    """
    with gr.Tab("⚙️ Settings & System", id="tab_settings") as tab:
        gr.Markdown(
            """
            ### ⚙️ System Configuration & Zero-PHI Compliance Shield
            *Configure API access keys, monitor Hugging Face ZeroGPU hardware acceleration, and verify privacy boundaries.*
            """
        )

        import importlib.util

        has_zerogpu = importlib.util.find_spec("spaces") is not None

        gpu_badge = (
            "<span style='background:#dcfce7;color:#166534;padding:4px 10px;border-radius:999px;font-weight:600;'>🟢 ZeroGPU Available (@spaces.GPU active)</span>"
            if has_zerogpu
            else "<span style='background:#f1f5f9;color:#475569;padding:4px 10px;border-radius:999px;'>⚪ CPU Host / Local Development Mode</span>"
        )

        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("##### 🔑 API Credentials & Cloud Inference")
                ncbi_input = gr.Textbox(
                    label="NCBI API Key (Optional — for PubMed E-Utilities):",
                    placeholder="Enter key to increase rate-limit to 10 req/sec...",
                    value=os.getenv("NCBI_API_KEY", ""),
                    type="password",
                )
                hf_input = gr.Textbox(
                    label="Hugging Face User Access Token (HF_TOKEN):",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                    value=os.getenv("HF_TOKEN", ""),
                    type="password",
                )
                with gr.Row():
                    btn_save = gr.Button("💾 Save Configuration", variant="primary")
                    btn_test_hf = gr.Button(
                        "🧪 Test Hugging Face Connection", variant="secondary"
                    )
                status_html = gr.HTML("")

            with gr.Column(scale=6):
                gr.Markdown("##### 🛡️ Zero-PHI Trust Boundary & Specifications")
                gr.HTML(
                    f"""
                    <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;'>
                        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
                            <strong style='color:#0f172a;'>Hardware & Acceleration:</strong>
                            {gpu_badge}
                        </div>
                        <ul style='color:#334155;font-size:0.88rem;line-height:1.7;margin:0;padding-left:20px;'>
                            <li><strong>Framework:</strong> Gradio 6.x Native Blocks + smolagents</li>
                            <li><strong>Zero-PHI Policy:</strong> Pure client/local sanitization before transmission (HMAC surrogate IDs & $T_0=0$ relative elapsed days).</li>
                            <li><strong>Statistical Benchmark:</strong> R 4.3.3 <code>survival::coxph</code> (Efron ties) & statsmodels 0.14+ GLM parity.</li>
                            <li><strong>Reporting Compliance:</strong> SAMPL, ICMJE, STROBE, CONSORT, TRIPOD+AI, STARD, PRISMA 2020.</li>
                        </ul>
                    </div>
                    """
                )

        btn_save.click(
            fn=update_settings_action,
            inputs=[ncbi_input, hf_input, app_state],
            outputs=[status_html],
        )

        btn_test_hf.click(
            fn=test_hf_connection_action,
            inputs=[hf_input, app_state],
            outputs=[status_html],
        )

    return tab, {
        "ncbi_input": ncbi_input,
        "hf_input": hf_input,
        "btn_save": btn_save,
        "btn_test_hf": btn_test_hf,
    }
