from ml.station_truth import daily


def _rows(day, temps, rh=50.0, wind=4.0):
    return [{"valid": f"{day} {h:02d}:53", "tmpf": str(t), "relh": str(rh), "sknt": str(wind)}
            for h, t in enumerate(temps)]


def test_daily_means_spike_and_flatline():
    temps = [50 + (h % 5) for h in range(24)]
    temps[10] = 120  # one-reading spike: dropped, not averaged in
    d = daily(_rows("2020-07-01", temps))["2020-07-01"]
    assert d["qc_dropped"] == 1 and d["n_temp"] == 23
    assert abs(d["temperature_celsius"] - ((sum(temps) - 120) / 23 - 32) / 1.8) < 1e-3
    assert d["humidity"] == ""              # constant all day: stuck sensor
    assert d["wind_speed"] == round(4 * 0.5144, 3)  # constant wind is allowed (calm is real)


def test_daily_needs_enough_readings():
    d = daily(_rows("2020-07-01", [60.0 + h for h in range(10)]))["2020-07-01"]
    assert d["temperature_celsius"] == "" and d["n_temp"] == 10
