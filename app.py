import gradio as gr
import tensorflow as tf
import numpy as np

MODEL_PATH = "models/mlp_dwp_plus_aux_best.keras"
model = tf.keras.models.load_model(MODEL_PATH)

ROOF_OPTS = ["outdoors", "dome", "closed", "retractable"]
SURFACE_OPTS = ["grass", "turf"]


def predict(
    quarter,
    time_left,
    yardline_100,
    ydstogo,
    score_diff,
    receive_2h_ko,
    timeouts_off,
    timeouts_def,
    roof,
    surface,
    temp_f,
    wind_mph,
    spread_line,
):
    # --- convert inputs ---
    mins, secs = map(int, time_left.split(":"))
    time_left_qtr = mins * 60 + secs
    game_sec = (4 - quarter) * 900 + time_left_qtr
    half_sec = ((2 if quarter <= 2 else 4) - quarter) * 900 + time_left_qtr
    fg_dist = yardline_100 + 17

    # derived features
    in_fg_range = float(fg_dist <= 60)
    one_score = float(abs(score_diff) <= 8)
    late_game = float(game_sec < 600)
    second_half = float(quarter > 2)
    timeouts_tot = timeouts_off + timeouts_def
    timeouts_dif = timeouts_off - timeouts_def

    X = {
        "quarter": float(quarter),
        "game_seconds_remaining": float(game_sec),
        "half_seconds_remaining": float(half_sec),
        "yardline_100": float(yardline_100),
        "ydstogo": float(ydstogo),
        "score_differential": float(score_diff),
        "receive_2h_ko": float(receive_2h_ko),
        "timeouts_off": float(timeouts_off),
        "timeouts_def": float(timeouts_def),
        "timeouts_total": float(timeouts_tot),
        "timeouts_diff": float(timeouts_dif),
        "temp_f": float(temp_f),
        "wind_mph": float(wind_mph),
        "spread_line": float(spread_line),
        "fg_dist_yd": float(fg_dist),
        "in_fg_range": float(in_fg_range),
        "one_score": float(one_score),
        "late_game": float(late_game),
        "second_half": float(second_half),
        "roof": roof,
        "surface": surface,
    }

    pred = model.predict({k: np.array([v]) for k, v in X.items()}, verbose=0)["dwp"][0]
    actions = ["Go for it", "Field goal", "Punt"]
    best = actions[np.argmax(pred)]
    return {a: float(p) for a, p in zip(actions, pred)}, best


with gr.Blocks() as demo:
    gr.Markdown("# NFL 4th‑down Decision Helper")
    with gr.Row():
        quarter = gr.Dropdown([1, 2, 3, 4], label="Quarter")
        time_left = gr.Textbox("12:00", label="Time left (MM:SS)")
        yardline_100 = gr.Slider(0, 100, label="Yardline (to goal)")
        ydstogo = gr.Slider(1, 25, label="Yards to go")
        score_diff = gr.Slider(
            -50, 50, 0, label="Score differential (offense minus defense)"
        )
    with gr.Row():
        receive_2h = gr.Checkbox(label="Receiving 2nd-half kickoff?")
        to_off = gr.Slider(0, 3, 3, label="Off. timeouts")
        to_def = gr.Slider(0, 3, 3, label="Def. timeouts")
    with gr.Row():
        roof = gr.Dropdown(ROOF_OPTS, label="Roof")
        surface = gr.Dropdown(SURFACE_OPTS, label="Surface")
        temp = gr.Slider(-10, 120, 70, label="Temperature (°F)")
        wind = gr.Slider(0, 50, 5, label="Wind (mph)")
        spread = gr.Slider(-30, 30, 0, label="Point spread")

    btn = gr.Button("Recommend")
    out_probs = gr.Label(num_top_classes=3, label="Δ Win Probability")
    out_best = gr.Textbox(label="Best choice", interactive=False)

    btn.click(
        predict,
        inputs=[
            quarter,
            time_left,
            yardline_100,
            ydstogo,
            score_diff,
            receive_2h,
            to_off,
            to_def,
            roof,
            surface,
            temp,
            wind,
            spread,
        ],
        outputs=[out_probs, out_best],
    )

if __name__ == "__main__":
    demo.launch()
