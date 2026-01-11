#!/usr/bin/env python3
"""
Check for thermal throttling on the system.

Monitors CPU temperatures, frequencies, and throttling events.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_cpu_temps() -> Dict[str, float]:
    """Get CPU temperatures from thermal zones."""
    temps = {}
    thermal_base = Path("/sys/class/thermal")

    if not thermal_base.exists():
        return temps

    for zone in thermal_base.glob("thermal_zone*"):
        zone_type = (zone / "type").read_text().strip()
        try:
            temp = (
                int((zone / "temp").read_text().strip()) / 1000.0
            )  # Convert from millidegrees
            temps[zone_type] = temp
        except (ValueError, FileNotFoundError):
            continue

    return temps


def get_cpu_frequencies() -> List[Optional[int]]:
    """Get current CPU frequencies in MHz."""
    frequencies = []
    cpu_base = Path("/sys/devices/system/cpu")

    for cpu_dir in sorted(cpu_base.glob("cpu[0-9]*")):
        freq_file = cpu_dir / "cpufreq" / "scaling_cur_freq"
        if freq_file.exists():
            try:
                freq_khz = int(freq_file.read_text().strip())
                frequencies.append(freq_khz // 1000)  # Convert to MHz
            except (ValueError, FileNotFoundError):
                frequencies.append(None)
        else:
            frequencies.append(None)

    return frequencies


def get_max_frequency() -> Optional[int]:
    """Get maximum CPU frequency in MHz."""
    cpu0_max = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
    if cpu0_max.exists():
        try:
            freq_khz = int(cpu0_max.read_text().strip())
            return freq_khz // 1000
        except (ValueError, FileNotFoundError):
            pass
    return None


def get_min_frequency() -> Optional[int]:
    """Get minimum CPU frequency in MHz."""
    cpu0_min = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq")
    if cpu0_min.exists():
        try:
            freq_khz = int(cpu0_min.read_text().strip())
            return freq_khz // 1000
        except (ValueError, FileNotFoundError):
            pass
    return None


def check_throttling_events() -> Dict[str, int]:
    """Check for throttling events in thermal zones."""
    events = {}
    thermal_base = Path("/sys/class/thermal")

    if not thermal_base.exists():
        return events

    for zone in thermal_base.glob("thermal_zone*"):
        zone_name = zone.name
        throttled_file = zone / "throttle0_total"
        if throttled_file.exists():
            try:
                count = int(throttled_file.read_text().strip())
                if count > 0:
                    events[zone_name] = count
            except (ValueError, FileNotFoundError):
                continue

    return events


def get_governor() -> Optional[str]:
    """Get CPU frequency scaling governor."""
    gov_file = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov_file.exists():
        return gov_file.read_text().strip()
    return None


def get_nvidia_gpu_info() -> Optional[Dict]:
    """Get NVIDIA GPU information using nvidia-smi."""
    try:
        # Get basic info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.max.memory,power.draw,power.limit,power.min_limit,power.max_limit",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        # Parse CSV output
        parts = result.stdout.strip().split(", ")
        if len(parts) < 8:
            return None

        name = parts[0]
        temp = float(parts[1])
        graphics_current = int(parts[2].replace(" MHz", ""))
        graphics_max = int(parts[3].replace(" MHz", ""))
        memory_current = int(parts[4].replace(" MHz", ""))
        memory_max = int(parts[5].replace(" MHz", ""))
        power_draw = float(parts[6].replace(" W", ""))
        power_limit = float(parts[7].replace(" W", ""))
        power_min = float(parts[8].replace(" W", "")) if len(parts) > 8 else None
        power_max = float(parts[9].replace(" W", "")) if len(parts) > 9 else None

        # Get throttling info
        throttling_info = {}
        try:
            q_result = subprocess.run(
                ["nvidia-smi", "-q", "-d", "PERFORMANCE"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if q_result.returncode == 0:
                output = q_result.stdout
                # Check for thermal slowdown
                if "HW Thermal Slowdown" in output:
                    for line in output.split("\n"):
                        if "HW Thermal Slowdown" in line:
                            throttling_info["hw_thermal"] = "Active" in line
                        elif "SW Thermal Slowdown" in line:
                            throttling_info["sw_thermal"] = "Active" in line
                        elif "Performance State" in line:
                            throttling_info["perf_state"] = line.split(":")[-1].strip()
        except Exception:
            pass

        # Get temperature limits
        temp_limits = {}
        try:
            t_result = subprocess.run(
                ["nvidia-smi", "-q", "-d", "TEMPERATURE"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if t_result.returncode == 0:
                output = t_result.stdout
                for line in output.split("\n"):
                    if "Slowdown Temp" in line:
                        temp_limits["slowdown"] = float(
                            line.split(":")[-1].strip().replace(" C", "")
                        )
                    elif "Shutdown Temp" in line:
                        temp_limits["shutdown"] = float(
                            line.split(":")[-1].strip().replace(" C", "")
                        )
                    elif "Target Temperature" in line:
                        temp_limits["target"] = float(
                            line.split(":")[-1].strip().replace(" C", "")
                        )
        except Exception:
            pass

        # Get persistence mode
        persistence_mode = None
        try:
            p_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=persistence_mode", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if p_result.returncode == 0:
                persistence_mode = p_result.stdout.strip()
        except Exception:
            pass

        return {
            "name": name,
            "temperature": temp,
            "graphics_current": graphics_current,
            "graphics_max": graphics_max,
            "memory_current": memory_current,
            "memory_max": memory_max,
            "power_draw": power_draw,
            "power_limit": power_limit,
            "power_min": power_min,
            "power_max": power_max,
            "throttling": throttling_info,
            "temp_limits": temp_limits,
            "persistence_mode": persistence_mode,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        return None


def get_amd_gpu_info() -> Optional[Dict]:
    """Get AMD GPU information from sysfs."""
    # Try to find AMD GPU hwmon
    for hwmon in Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*"):
        temp_file = hwmon / "temp1_input"
        if temp_file.exists():
            try:
                temp = int(temp_file.read_text().strip()) / 1000.0
                return {
                    "name": "AMD GPU",
                    "temperature": temp,
                    "source": "sysfs",
                }
            except (ValueError, FileNotFoundError):
                continue
    return None


def main():
    print("=" * 60)
    print("Thermal Throttling Check")
    print("=" * 60)

    # Get temperatures
    temps = get_cpu_temps()
    if temps:
        print("\n📊 CPU Temperatures:")
        for zone_type, temp in sorted(temps.items()):
            status = "🟢 OK" if temp < 80 else "🟡 WARM" if temp < 90 else "🔴 HOT"
            print(f"  {zone_type:20s}: {temp:6.1f}°C {status}")
    else:
        print("\n⚠️  Could not read CPU temperatures")

    # Get frequencies
    frequencies = get_cpu_frequencies()
    max_freq = get_max_frequency()
    min_freq = get_min_frequency()
    governor = get_governor()

    if frequencies:
        active_freqs = [f for f in frequencies if f is not None]
        if active_freqs:
            avg_freq = sum(active_freqs) / len(active_freqs)
            min_active = min(active_freqs)
            max_active = max(active_freqs)

            print(f"\n⚡ CPU Frequencies:")
            print(f"  Active cores: {len(active_freqs)}/{len(frequencies)}")
            print(f"  Current range: {min_active:.0f} - {max_active:.0f} MHz")
            print(f"  Average: {avg_freq:.0f} MHz")
            if max_freq:
                print(f"  Max frequency: {max_freq:.0f} MHz")
            if min_freq:
                print(f"  Min frequency: {min_freq:.0f} MHz")
            if governor:
                print(f"  Governor: {governor}")

            # Check if throttling might be happening
            if max_freq and max_active < max_freq * 0.7:
                print(f"\n⚠️  WARNING: CPUs running significantly below max frequency!")
                print(f"   This could indicate throttling (thermal, power, or other)")
            elif min_active == min_freq and max_active < max_freq * 0.5:
                print(f"\n💤 CPUs appear to be in idle/powersave mode")

    # Check throttling events
    throttling_events = check_throttling_events()
    if throttling_events:
        print(f"\n🚨 CPU THROTTLING EVENTS DETECTED:")
        for zone, count in throttling_events.items():
            print(f"  {zone}: {count} throttling events")
    else:
        print(f"\n✅ No CPU throttling events detected")

    # Check GPU
    gpu_throttling = False
    nvidia_gpu = get_nvidia_gpu_info()
    if nvidia_gpu:
        print(f"\n🎮 GPU: {nvidia_gpu['name']}")
        print(f"  Temperature: {nvidia_gpu['temperature']:.1f}°C", end="")

        # Check temperature status
        temp_limits = nvidia_gpu.get("temp_limits", {})
        slowdown_temp = temp_limits.get("slowdown", 90)
        target_temp = temp_limits.get("target", 83)

        if nvidia_gpu["temperature"] >= slowdown_temp:
            print(" 🔴 HOT (near slowdown)")
            gpu_throttling = True
        elif nvidia_gpu["temperature"] >= target_temp:
            print(" 🟡 WARM")
        else:
            print(" 🟢 OK")

        if temp_limits:
            print(
                f"  Target temp: {target_temp:.0f}°C | Slowdown: {slowdown_temp:.0f}°C"
            )

        # Clock information
        graphics_pct = (
            (nvidia_gpu["graphics_current"] / nvidia_gpu["graphics_max"] * 100)
            if nvidia_gpu["graphics_max"] > 0
            else 0
        )
        memory_pct = (
            (nvidia_gpu["memory_current"] / nvidia_gpu["memory_max"] * 100)
            if nvidia_gpu["memory_max"] > 0
            else 0
        )

        print(
            f"  Graphics clock: {nvidia_gpu['graphics_current']} / {nvidia_gpu['graphics_max']} MHz ({graphics_pct:.0f}%)"
        )
        print(
            f"  Memory clock: {nvidia_gpu['memory_current']} / {nvidia_gpu['memory_max']} MHz ({memory_pct:.0f}%)"
        )

        # Power information
        power_pct = (
            (nvidia_gpu["power_draw"] / nvidia_gpu["power_limit"] * 100)
            if nvidia_gpu["power_limit"] > 0
            else 0
        )
        print(
            f"  Power: {nvidia_gpu['power_draw']:.1f} / {nvidia_gpu['power_limit']:.1f} W ({power_pct:.0f}%)"
        )

        # Throttling status
        throttling = nvidia_gpu.get("throttling", {})
        if throttling.get("hw_thermal"):
            print(f"  🚨 HW Thermal Slowdown: ACTIVE")
            gpu_throttling = True
        elif throttling.get("sw_thermal"):
            print(f"  🚨 SW Thermal Slowdown: ACTIVE")
            gpu_throttling = True
        else:
            print(f"  ✅ No thermal throttling detected")

        if throttling.get("perf_state"):
            perf_state = throttling["perf_state"]
            # P0 = max performance, P8 = idle, P12 = minimum
            state_desc = {
                "P0": "Maximum Performance",
                "P1": "High Performance",
                "P2": "Balanced",
                "P3": "Low Power",
                "P8": "Idle",
                "P12": "Minimum Power",
            }
            desc = state_desc.get(perf_state, "Unknown")
            print(f"  Performance state: {perf_state} ({desc})")

            if perf_state not in ["P0", "P1"]:
                print(f"  💡 Tip: GPU is in {perf_state} state (not max performance)")
                print(f"     Performance states are auto-managed by the driver.")
                print(f"     To influence performance:")
                if nvidia_gpu.get("power_max"):
                    print(
                        f"       - Increase power limit: sudo nvidia-smi -pl {nvidia_gpu['power_max']:.0f}"
                    )
                print(f"       - Enable persistence mode: sudo nvidia-smi -pm 1")
                print(f"       - Run GPU workload to boost to P0/P1")

        # Show power limit range if available
        if nvidia_gpu.get("power_min") and nvidia_gpu.get("power_max"):
            print(
                f"  Power limit range: {nvidia_gpu['power_min']:.0f} - {nvidia_gpu['power_max']:.0f} W"
            )
            if nvidia_gpu["power_limit"] < nvidia_gpu["power_max"]:
                print(
                    f"  💡 Power limit is below max - increasing may improve performance"
                )

        # Show persistence mode
        if nvidia_gpu.get("persistence_mode"):
            pm_status = (
                "✅ Enabled"
                if nvidia_gpu["persistence_mode"] == "Enabled"
                else "❌ Disabled"
            )
            print(f"  Persistence mode: {pm_status}")
            if nvidia_gpu["persistence_mode"] != "Enabled":
                print(f"  💡 Enable with: sudo nvidia-smi -pm 1")

        # Check if clocks are significantly below max (potential throttling)
        if graphics_pct < 80 and nvidia_gpu["temperature"] > target_temp:
            print(f"  ⚠️  Graphics clock below max - possible throttling")

    else:
        # Try AMD GPU
        amd_gpu = get_amd_gpu_info()
        if amd_gpu:
            print(f"\n🎮 GPU: {amd_gpu['name']}")
            print(f"  Temperature: {amd_gpu['temperature']:.1f}°C", end="")
            if amd_gpu["temperature"] > 90:
                print(" 🔴 HOT")
                gpu_throttling = True
            elif amd_gpu["temperature"] > 80:
                print(" 🟡 WARM")
            else:
                print(" 🟢 OK")
        else:
            print(f"\n🎮 GPU: Not detected or not accessible")

    # Summary
    print("\n" + "=" * 60)
    if throttling_events or gpu_throttling:
        print("❌ THERMAL THROTTLING IS OCCURRING")
        sys.exit(1)
    elif temps and any(t > 90 for t in temps.values()):
        print("⚠️  High temperatures detected - monitor closely")
        sys.exit(0)
    else:
        print("✅ System appears healthy - no throttling detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
