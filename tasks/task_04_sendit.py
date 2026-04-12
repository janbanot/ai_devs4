import json
import logging
import math
import re
import sys
from datetime import date
from pathlib import Path

import dspy
import httpx

from ai_devs import Config, configure_logging, fetch_data, submit_answer

logger = logging.getLogger(__name__)

DOCS_DIR = Path("data/docs")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


class SendItAgent:
    def __init__(self, config: Config):
        self.config = config
        self.docs_content: dict[str, str] = {}
        self.plan: dict[str, dict] = {}
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self._setup_tools()

    def _file_type(self, file_name: str) -> str:
        ext = Path(file_name).suffix.lower()
        return "image" if ext in IMAGE_EXTENSIONS else "text"

    def _add_to_plan(self, file_name: str) -> None:
        if file_name not in self.plan:
            self.plan[file_name] = {
                "status": "pending",
                "type": self._file_type(file_name),
            }
            logger.debug(
                "Added to plan: %s (%s)", file_name, self._file_type(file_name)
            )

    def _extract_includes_from_content(self, content: str) -> list[str]:
        pattern = r'\[include\s+file="([^"]+)"\]'
        matches = re.findall(pattern, content)
        for f in matches:
            self._add_to_plan(f)
        if matches:
            logger.info("Auto-discovered %d includes: %s", len(matches), matches)
        return matches

    def _setup_tools(self):
        docs_content = self.docs_content
        plan = self.plan
        config = self.config

        def download_doc(file_name: str) -> str:
            """Download a documentation file from the remote server. Saves it locally. For text files, also auto-discovers [include file="..."] references. Returns a summary with file info and any discovered includes."""
            try:
                local_path = DOCS_DIR / file_name
                file_type = self._file_type(file_name)
                self._add_to_plan(file_name)

                if not local_path.exists():
                    data = fetch_data(f"dane/doc/{file_name}", config)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    if file_type == "image":
                        local_path.write_bytes(data)
                    else:
                        local_path.write_text(data.decode("utf-8"), encoding="utf-8")

                plan[file_name]["status"] = "downloaded"
                if file_type == "image":
                    logger.info("Ready: %s (image)", file_name)
                    return f"Ready: {file_name} (image)"

                content = local_path.read_text(encoding="utf-8")
                includes = self._extract_includes_from_content(content)
                logger.info("Ready: %s (%d chars)", file_name, len(content))
                summary = f"Downloaded {file_name} ({len(content)} chars)."
                if includes:
                    summary += f" Discovered includes: {json.dumps(includes)}. Download and read them next."
                return summary
            except Exception as e:
                logger.error("download_doc error for %s: %s", file_name, e)
                return f"Error downloading {file_name}: {e}"

        def read_doc(file_name: str) -> str:
            """Read a previously downloaded text document. Stores content internally and auto-discovers [include file="..."] references. Returns a summary with discovered includes."""
            try:
                local_path = DOCS_DIR / file_name
                if not local_path.exists():
                    return (
                        f"File not found locally: {file_name}. Call download_doc first."
                    )
                content = local_path.read_text(encoding="utf-8")
                self._add_to_plan(file_name)
                plan[file_name]["status"] = "read"
                docs_content[file_name] = content
                includes = self._extract_includes_from_content(content)
                logger.info("Read: %s (%d chars)", file_name, len(content))
                summary = f"Read {file_name} ({len(content)} chars)."
                if includes:
                    summary += f" Discovered includes: {json.dumps(includes)}. Download and read them next."
                return summary
            except Exception as e:
                logger.error("read_doc error for %s: %s", file_name, e)
                return f"Error reading {file_name}: {e}"

        def read_image(file_name: str) -> str:
            """Read and describe a previously downloaded image using vision AI. Stores the description internally. Returns a brief summary."""
            try:
                local_path = DOCS_DIR / file_name
                if not local_path.exists():
                    return f"Image not found locally: {file_name}. Call download_doc first."

                image = dspy.Image(str(local_path))
                describe = dspy.Predict(
                    "image: dspy.Image, prompt: str -> description: str"
                )
                result = describe(
                    image=image,
                    prompt=(
                        "You are analyzing an image that is part of a technical documentation. "
                        "Extract and describe ALL information visible in this image: "
                        "any text, numbers, table data, labels, legends, diagrams, maps, "
                        "or any other meaningful content. Be thorough and precise. "
                        "Reproduce any tables in text format. List any listed items exactly."
                    ),
                )
                description = result.description
                self._add_to_plan(file_name)
                plan[file_name]["status"] = "read"
                docs_content[file_name] = description
                logger.info("Read image: %s (%d chars)", file_name, len(description))
                return f"Read image {file_name} ({len(description)} chars)."
            except Exception as e:
                logger.error("read_image error for %s: %s", file_name, e)
                return f"Error reading image {file_name}: {e}"

        def extract_includes(file_name: str) -> str:
            """Scan a downloaded text file for [include file="..."] references without fully reading it. Returns JSON array of discovered filenames and adds them to the plan as pending."""
            try:
                local_path = DOCS_DIR / file_name
                if not local_path.exists():
                    return (
                        f"File not found locally: {file_name}. Call download_doc first."
                    )
                content = local_path.read_text(encoding="utf-8")
                includes = self._extract_includes_from_content(content)
                return json.dumps(includes)
            except Exception as e:
                logger.error("extract_includes error: %s", e)
                return f"Error: {e}"

        def get_plan() -> str:
            """Get the current plan showing all files, their types and statuses, and which are still pending. Returns JSON."""
            pending = [f for f, info in plan.items() if info["status"] != "read"]
            summary = {
                "total": len(plan),
                "read": len(
                    [f for f, info in plan.items() if info["status"] == "read"]
                ),
                "pending": pending,
                "details": {f: info["status"] for f, info in sorted(plan.items())},
            }
            logger.debug(
                "Plan: %d total, %d read, %d pending",
                summary["total"],
                summary["read"],
                len(pending),
            )
            return json.dumps(summary)

        def plan_route(origin: str, destination: str) -> str:
            """Extract the excluded route code from the image description (trasy-wylaczone.png). The image contains the X-code for the direct excluded route between cities. Returns JSON with route code and details."""
            try:
                origin_lower = origin.strip().lower()
                dest_lower = destination.strip().lower()

                x_code_pattern = re.compile(r"X-\d+")
                city_pair_pattern = re.compile(
                    r"(X-\d+)\s*[:\-–—.]?\s*"
                    r"([A-ZĄĆĘŁŃÓŚŹŻ][\wĄĆĘŁŃÓŚŹŻ\-]*)\s*[-–—]\s*"
                    r"([A-ZĄĆĘŁŃÓŚŹŻ][\wĄĆĘŁŃÓŚŹŻ\-]*)",
                    re.IGNORECASE,
                )

                for fname, content in docs_content.items():
                    if not (fname.endswith(".png") or "trasy" in fname):
                        continue
                    for match in city_pair_pattern.finditer(content):
                        code = match.group(1).strip()
                        c1 = match.group(2).strip().lower()
                        c2 = match.group(3).strip().lower()
                        if {c1, c2} == {origin_lower, dest_lower}:
                            logger.info(
                                "Found excluded route: %s (%s <-> %s)",
                                code,
                                match.group(2).strip(),
                                match.group(3).strip(),
                            )
                            return json.dumps(
                                {
                                    "route_code": code,
                                    "from": origin.strip(),
                                    "to": destination.strip(),
                                }
                            )

                    for match in x_code_pattern.finditer(content):
                        code = match.group(0)
                        logger.info(
                            "Found X-code in %s: %s (using as direct route)",
                            fname,
                            code,
                        )
                        return json.dumps(
                            {
                                "route_code": code,
                                "from": origin.strip(),
                                "to": destination.strip(),
                            }
                        )

                return json.dumps(
                    {"error": f"No X-route code found for {origin} -> {destination}"}
                )
            except Exception as e:
                logger.error("plan_route error: %s", e)
                return json.dumps({"error": str(e)})

        self.tools = [
            download_doc,
            read_doc,
            read_image,
            extract_includes,
            get_plan,
            plan_route,
        ]

    def get_all_content(self) -> str:
        combined = []
        for file_name, content in sorted(self.docs_content.items()):
            combined.append(f"=== {file_name} ===\n{content}")
        return "\n\n".join(combined)


class GatherDocs(dspy.Signature):
    """Gather all documentation by downloading and reading referenced files, including images.
    Follow the plan systematically until all documents are read."""

    question: str = dspy.InputField()
    result: str = dspy.OutputField(
        desc="Comprehensive summary of all gathered documentation content"
    )


class GenerateDeclaration(dspy.Signature):
    """Generate a shipment declaration EXACTLY following the template from the documentation (Załącznik E).

    CRITICAL RULES:
    - Copy the template structure EXACTLY: same headers, same separators (===== and -----), same field names
    - Replace EVERY [placeholder] with an actual value - NO placeholders may remain
    - The DATA field MUST be today's date in YYYY-MM-DD format
    - The KWOTA DO ZAPŁATY for Category A is 0 (zero) because Cat A is fully funded by System
    - WDP = number of additional wagons (0 for single shipment)
    - Do NOT add any fields not in the template
    - Do NOT add signature lines or extra text after the closing separator
    - WDP (Wagony Dodatkowe Płatne): base train = 2 wagons × 500 kg = 1000 kg capacity. WDP = ceil((weight_kg - 1000) / 500) if weight > 1000, else 0. Each additional wagon adds 500 kg capacity.
    - For Category A shipments: KWOTA DO ZAPŁATY = 0, and WDP opłata is also 0 (covered by System), but still declare the correct WDP count."""

    documentation: str = dspy.InputField(
        desc="All gathered documentation including the declaration template from Załącznik E"
    )
    shipment_data: str = dspy.InputField(
        desc="Shipment details to fill into the declaration"
    )
    route_plan: str = dspy.InputField(
        desc="Planned route with segment codes and distances"
    )
    today_date: str = dspy.InputField(
        desc="Today's date in YYYY-MM-DD format - MUST be used for the DATA field"
    )
    declaration: str = dspy.OutputField(
        desc="Complete declaration text in the EXACT format from the documentation template, with all placeholders replaced"
    )


SHIPMENT_DATA = """\
Nadawca (identyfikator): 450202122
Punkt nadawczy: Gdańsk
Punkt docelowy: Żarnowiec
Waga: 2800 kg (2,8 tony)
Budżet: 0 PP (przesyłka ma być darmowa lub finansowana przez System)
Zawartość: kasety z paliwem do reaktora
Uwagi specjalne: brak - nie dodawaj żadnych uwag
"""


def main() -> None:
    configure_logging()
    config = Config()

    try:
        lm = dspy.LM(
            model="mistral/mistral-medium-latest", api_key=config.mistral_api_key
        )
        dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

        agent = SendItAgent(config)

        react = dspy.ReAct(
            GatherDocs,
            tools=agent.tools,
            max_iters=30,
        )

        question = """Gather all documentation by following these steps:

1. Call download_doc("index.md") to download the main index. It will auto-discover included files.
2. Call read_doc("index.md") to read it into memory. It will also discover includes.
3. For every discovered file reported by download_doc or read_doc:
    a. Call download_doc(file_name) to download it
    b. If text/markdown (.md, .txt): call read_doc(file_name) to read it (will also discover more includes)
    c. If image (.png, .jpg, .gif): call read_image(file_name) to analyze with vision
4. Call get_plan() to check if there are still pending/unread files
5. Keep downloading and reading until get_plan() shows 0 pending files
6. Once all documents are gathered, call plan_route("Gdańsk", "Żarnowiec") to find a valid route
7. Return a comprehensive summary of all documentation AND the planned route

IMPORTANT: Every download_doc and read_doc call auto-discovers [include file="..."] references.
Pay attention to the "Discovered includes" in each response - those are files you must also download and read.
Be thorough - download and read EVERY referenced file.
The route planning is critical - we need to find a connected path through the SPK network from Gdańsk to Żarnowiec.
The trasy-wylaczone.png image contains excluded routes that are LEGAL for Category A shipments."""

        logger.info("Running ReAct agent...")
        result = react(question=question)

        logger.info("Agent result: %s", result.result)

        all_content = agent.get_all_content()
        logger.info(
            "Gathered %d documents (%d total chars)",
            len(agent.docs_content),
            len(all_content),
        )

        for f, info in sorted(agent.plan.items()):
            logger.info("  %s [%s] %s", f, info["type"], info["status"])

        logger.info("Planning route from Gdańsk to Żarnowiec...")
        route_result = None
        for tool in agent.tools:
            if tool.__name__ == "plan_route":
                route_result = tool("Gdańsk", "Żarnowiec")
                break
        logger.info("Route plan: %s", route_result)

        logger.info("Generating declaration...")
        today = date.today().isoformat()
        generate = dspy.Predict(GenerateDeclaration)
        declaration_result = generate(
            documentation=all_content,
            shipment_data=SHIPMENT_DATA,
            route_plan=route_result or "{}",
            today_date=today,
        )
        declaration = declaration_result.declaration
        wdp = math.ceil(max(0, (2800 - 1000)) / 500)
        declaration = re.sub(r"WDP:\s*\d+", f"WDP: {wdp}", declaration)
        logger.info("Declaration generated (WDP=%d):\n%s", wdp, declaration)

        logger.info("Submitting answer...")
        response = submit_answer("sendit", {"declaration": declaration}, config)
        logger.info("Response: %s", response)

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error: %s", e)
        logger.error("Response body: %s", e.response.text)
        sys.exit(1)
    except httpx.HTTPError as e:
        logger.error("HTTP error: %s", e)
        sys.exit(1)
    except OSError as e:
        logger.error("IO error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
