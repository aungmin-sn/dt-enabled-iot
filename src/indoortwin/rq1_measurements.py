from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


from indoortwin.paths import ROOT
OUT = ROOT / 'results' / 'rq1_figures'
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = [
    ("ENV-IN-01", "ROOM01", "co2_indoor", "Indoor CO₂ (ppm)"),
    ("ENV-OUT-01", "OUTDOOR-REF", "co2", "Outdoor CO₂ (ppm)"),
    ("OCC-IN-01", "ROOM01", "occupancy", "Occupancy state"),
    ("HEAT-MON-01", "HEATER-01", "current", "Heater current (A)"),
    ("VIB-FAN-01", "FAN-EXH", "vibration_rms", "Exhaust-fan vibration RMS (g)"),
    ("VENT-CTRL-01", "ROOM01", "delta_pressure", "Room differential pressure (Pa)"),
]

NODE_COLORS = {
    "ENV-IN-01": "#EFCB68",
    "ENV-OUT-01": "#A7C957",
    "OCC-IN-01": "#52B69A",
    "HEAT-MON-01": "#7EA8BE",
    "VIB-FAN-01": "#B8A1D9",
    "VENT-CTRL-01": "#E79A9A",
}

STATE_MAP = {
    "ABSENT": 0,
    "UNOCCUPIED": 0,
    "FALSE": 0,
    "OFF": 0,
    "PRESENT": 1,
    "OCCUPIED": 1,
    "TRUE": 1,
    "ON": 1,
}


def font(size: int, bold: bool = False):
    from matplotlib import font_manager
    name = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(name, size=size)


def local_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(ZoneInfo("Pacific/Auckland"))


def as_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = series.astype(str).str.strip().str.upper().map(STATE_MAP)
    return numeric.fillna(mapped)


def text_width(draw: ImageDraw.ImageDraw, value: str, used_font) -> int:
    box = draw.textbbox((0, 0), value, font=used_font)
    return box[2] - box[0]


def make_plot(folder: Path, summary_name: str, title: str, output_name: str, date_label: str):
    received = pd.read_csv(folder / "received_values.csv")
    summary = pd.read_csv(folder / summary_name)
    received["time"] = local_time(received["received_time"])
    received["number"] = as_numeric(received["value"])
    summary["start"] = local_time(summary["disconnect_confirmed_time"])
    summary["end"] = local_time(summary["node_properties_fresh_time"])

    min_time = min(received["time"].min(), summary["start"].min()).to_pydatetime()
    max_time = max(received["time"].max(), summary["end"].max()).to_pydatetime()
    duration = (max_time - min_time).total_seconds()

    width, height = 3900, 3420
    left, right = 500, 3770
    first_top, panel_height, gap = 490, 360, 72
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(76, True)
    subtitle_font = font(46)
    label_font = font(56, True)
    tick_font = font(44)
    legend_font = font(44)
    axis_font = font(48)
    navy, grid, text, muted = "#1F5A82", "#D9DEE2", "#263238", "#5F6B73"

    draw.text((width / 2, 58), title, font=title_font, fill=text, anchor="ma")
    draw.text(
        (width / 2, 166),
        "Shading spans confirmed interruption to complete property-state restoration; lines break across observation gaps longer than 4 s",
        font=subtitle_font,
        fill=muted,
        anchor="ma",
    )

    draw.text((width / 2, 245), "Interrupted node", font=legend_font, fill=text, anchor="ma")
    legend_items = list(NODE_COLORS.items())
    for item_index, (node, colour) in enumerate(legend_items):
        row, column = divmod(item_index, 3)
        legend_x = 650 + column * 1100
        legend_y = 315 + row * 82
        draw.rounded_rectangle((legend_x, legend_y - 25, legend_x + 68, legend_y + 25), radius=7, fill=colour)
        draw.text((legend_x + 90, legend_y), node, font=legend_font, fill=text, anchor="lm")

    def sx(moment):
        return left + (moment - min_time).total_seconds() / duration * (right - left)

    for panel_index, (node, entity, prop, measure_label) in enumerate(CHANNELS):
        top = first_top + panel_index * (panel_height + gap)
        bottom = top + panel_height
        subset = received[
            (received["node_id"] == node)
            & (received["entity_id"] == entity)
            & (received["property"] == prop)
        ][["time", "number"]].dropna().sort_values("time")
        if subset.empty:
            raise ValueError(f"No data found for {node}/{entity}/{prop}")

        values = subset["number"].astype(float).tolist()
        low, high = min(values), max(values)
        if prop == "occupancy":
            low, high = -0.15, 1.15
        else:
            span = high - low
            pad = max(span * 0.11, abs(high) * 0.025, 0.02)
            low, high = low - pad, high + pad

        def sy(value):
            return bottom - (value - low) / (high - low) * (bottom - top)

        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for row in summary.itertuples(index=False):
            colour = NODE_COLORS.get(str(row.node_id), "#CCCCCC")
            rgb = tuple(int(colour[index:index + 2], 16) for index in (1, 3, 5))
            overlay_draw.rectangle((sx(row.start.to_pydatetime()), top, sx(row.end.to_pydatetime()), bottom), fill=rgb + (83,))
        image.paste(overlay, (0, 0), overlay)
        draw = ImageDraw.Draw(image)

        for grid_index in range(5):
            y = top + grid_index * (bottom - top) / 4
            draw.line((left, y, right, y), fill=grid, width=2)
            if prop == "occupancy":
                if grid_index in (0, 4):
                    tick_text = "Present" if grid_index == 0 else "Absent"
                    draw.text((left - 20, y), tick_text, font=tick_font, fill=text, anchor="rm")
            else:
                value = high - grid_index * (high - low) / 4
                precision = 2 if max(abs(low), abs(high)) < 10 else (1 if max(abs(low), abs(high)) < 100 else 0)
                draw.text((left - 20, y), f"{value:.{precision}f}", font=tick_font, fill=text, anchor="rm")
        draw.rectangle((left, top, right, bottom), outline="#7B858D", width=3)
        panel_label = f"{node} — {measure_label}"
        panel_label_width = text_width(draw, panel_label, label_font)

        previous_time = None
        previous_point = None
        for row in subset.itertuples(index=False):
            moment = row.time.to_pydatetime()
            point = (sx(moment), sy(float(row.number)))
            if previous_time is not None and (moment - previous_time).total_seconds() <= 4:
                if prop == "occupancy":
                    draw.line((previous_point[0], previous_point[1], point[0], previous_point[1]), fill=navy, width=7)
                    draw.line((point[0], previous_point[1], point[0], point[1]), fill=navy, width=7)
                else:
                    draw.line((previous_point[0], previous_point[1], point[0], point[1]), fill=navy, width=7)
            previous_time, previous_point = moment, point

        draw.rounded_rectangle(
            (left + 8, top + 7, left + panel_label_width + 35, top + 75),
            radius=8,
            fill="#FFFFFF",
        )
        draw.text((left + 18, top + 11), panel_label, font=label_font, fill=text, anchor="la")

    last_bottom = first_top + 5 * (panel_height + gap) + panel_height
    tick_seconds = 30 if duration <= 420 else 60
    first_epoch = math.ceil(min_time.timestamp() / tick_seconds) * tick_seconds
    last_epoch = math.floor(max_time.timestamp() / tick_seconds) * tick_seconds
    for epoch in range(first_epoch, last_epoch + 1, tick_seconds):
        moment = datetime.fromtimestamp(epoch, tz=ZoneInfo("Pacific/Auckland"))
        x = sx(moment)
        draw.line((x, last_bottom, x, last_bottom + 14), fill=text, width=3)
        draw.text((x, last_bottom + 25), moment.strftime("%H:%M:%S"), font=tick_font, fill=text, anchor="ma")
    draw.text((width / 2, height - 48), date_label, font=axis_font, fill=text, anchor="ma")
    image.save(OUT / output_name, dpi=(300, 300))
