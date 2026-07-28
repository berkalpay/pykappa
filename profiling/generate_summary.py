import os
import re
import json
import subprocess
from collections import defaultdict

if __name__ == "__main__":
    print("\n\nGenerating summary branch summary...")
    base_url = re.sub(
        r"git@github\.com:(.+?)\.git",
        r"https://github.com/\1",
        subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
        .decode()
        .strip(),
    )

    # 1. Get ordered history of the profiled source revision
    source_revision = os.environ.get("PROFILE_SOURCE_SHA", "HEAD")
    git_log = subprocess.check_output(
        ["git", "log", "--reverse", "--pretty=format:%h", source_revision], text=True
    ).splitlines()

    branch_name = (
        os.environ.get("PROFILE_SOURCE_REF")
        or subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip()
    )

    # 2. Identify relevant commit directories
    commit_dirs = {}
    dir_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}_commit_([a-f0-9]+)$")
    for d in os.listdir("."):
        if os.path.isdir(d):
            match = dir_pattern.match(d)
            if match:
                commit_hash = match.group(1)
                if os.path.exists(d):
                    commit_dirs[commit_hash] = d

    profiled_commits = [c for c in git_log if c in commit_dirs]
    print(f"{git_log=}")
    print(f"{commit_dirs=}")
    print(f"{profiled_commits=}")

    # 3. Extract profile data from summary files
    profile_runtimes = defaultdict(list)
    profile_memories = defaultdict(list)

    for commit in profiled_commits:
        with open(os.path.join(commit_dirs[commit], "summary.json")) as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                continue

        found_profiles = set()
        for key, data in summary.items():
            if key.startswith("profile_"):
                profile_name = key.replace("profile_", "")
                found_profiles.add(profile_name)

                # Add current data point
                profile_runtimes[profile_name].append(data["runtime (s)"])
                profile_memories[profile_name].append(data["peak_memory (MB)"])

        # Ensure all profiles have entries (even if missing in this commit)
        for profile in profile_runtimes.keys():
            if profile not in found_profiles:
                profile_runtimes[profile].append(None)
                profile_memories[profile].append(None)

    # 4. Generate HTML with Plotly visualizations
    # ... (previous code remains the same)

    # Precompute JavaScript data structures
    commit_hashes_json = json.dumps(profiled_commits)
    short_hashes = [h[:7] for h in profiled_commits]
    linked_hashes = [
        f'<a href="{base_url}/commit/{h}">{h[:7]}</a>' for h in profiled_commits
    ]

    # Prepare runtime data
    runtime_data = []
    for name, values in profile_runtimes.items():
        runtime_data.append(
            {
                "x": profiled_commits,
                "y": values,
                "name": name,
                "mode": "lines+markers",
                "connectgaps": False,
                "text": short_hashes,
                "customdata": [
                    # f"{base_url}/raw/{PROFILING_BRANCH}/{commit_dirs[h]}/profile_{name}_flamegraph.svg"
                    f"{commit_dirs[h]}/profile_{name}_flamegraph.svg"
                    for h in profiled_commits
                ],
                "hovertemplate": f"<b>%{{text}}</b><br><b>Runtime: %{{y:.4f}}s</b><br>(Click to open flamegraph)",
                "marker": {"size": 12},
            }
        )
    runtime_data_json = json.dumps(runtime_data)

    # Prepare memory data
    memory_data = []
    for name, values in profile_memories.items():
        memory_data.append(
            {
                "x": profiled_commits,
                "y": values,
                "name": name,
                "mode": "lines+markers",
                "connectgaps": False,
                "text": short_hashes,
                "customdata": [
                    f"{commit_dirs[h]}/profile_{name}_memplot.png"
                    for h in profiled_commits
                ],
                "hovertemplate": f"<b>%{{text}}</b><br><b>Peak Memory: %{{y:.2f}} MB</b><br>(Click to open memory usage plot)",
                "marker": {"size": 12},
            }
        )
    memory_data_json = json.dumps(memory_data)

    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{branch_name} performance log</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .plot-container {{ width: 100%; height: 600px; margin-bottom: 40px; }}
            /* Change cursor to pointer for plot points */
            .cursor-pointer .nsewdrag, .cursor-pointer .nsewdrag drag {{
                cursor: pointer !important;
            }}
        </style>
    </head>
    <body>
        <h1>Performance log, branch <a href="{base_url}/tree/{branch_name}">{branch_name}</a></h1>

        <div class="plot-container" id="runtimePlot"></div>
        <div class="plot-container" id="memoryPlot"></div>

        <script>
            // Shared configuration
            const commitHashes = {commit_hashes_json};
            const shortHashes = {json.dumps(linked_hashes)};
            const plotConfig = {{responsive: true}};

            // Runtime data and plot
            const runtimeData = {runtime_data_json};

            Plotly.newPlot('runtimePlot', runtimeData, {{
                title: 'Runtime (seconds)',
                titlefont: {{ size: 30, weight: 'bold' }},
                xaxis: {{
                    title: 'Commit Hash',
                    type: 'category',
                    tickangle: -45,
                    tickmode: 'array',
                    tickvals: commitHashes,
                    ticktext: shortHashes,
                }},
                yaxis: {{ title: 'Runtime (s)',
                        rangemode: 'tozero'}},
                hovermode: 'closest',
                showlegend: true
            }}, plotConfig);

            // Memory data and plot
            const memoryData = {memory_data_json};
            memoryData.forEach(trace => trace.marker = {{size: 12}});

            Plotly.newPlot('memoryPlot', memoryData, {{
                title: 'Peak Memory Usage (MB)',
                titlefont: {{ size: 30, weight: 'bold' }},
                xaxis: {{
                    title: 'Commit Hash',
                    type: 'category',
                    tickangle: -45,
                    tickmode: 'array',
                    tickvals: commitHashes,
                    ticktext: shortHashes,
                }},
                yaxis: {{ title: 'Peak Memory (MB)',
                        rangemode: 'tozero'}},
                hovermode: 'closest',
                showlegend: true
            }}, plotConfig);

            ['runtimePlot', 'memoryPlot'].forEach(plotId => {{
                const plot = document.getElementById(plotId);

                // Change cursor on hover
                plot.on('plotly_hover', function(data) {{
                    plot.classList.add('cursor-pointer');
                }});

                // Revert cursor when not hovering
                plot.on('plotly_unhover', function(data) {{
                    plot.classList.remove('cursor-pointer');
                }});

                // Click handler
                plot.on('plotly_click', function(data) {{
                    const url = data.points[0].customdata;
                    if (url) window.open(url, '_blank');
                }});
            }});
        </script>
    </body>
    </html>
    """

    # Save the HTML file
    filename = f"{branch_name}_branch_summary.html"
    with open(filename, "w") as f:
        f.write(html_content)

    print(f"Generated {filename}")
