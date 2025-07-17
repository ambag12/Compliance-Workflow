import sys
try:
    import pysqlite3 as sqlite3
    sys.modules["sqlite3"] = sqlite3
except ImportError:
    # If pysqlite3 isn't installed, we'll fallback to system sqlite3 (might still error)
    pass
import os
import re
import json
import csv
import yaml
from collections import defaultdict
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_core.messages import HumanMessage
from langchain_cohere import ChatCohere
from tavily import TavilyClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

if not all([os.getenv("COHERE_API_KEY"), os.getenv("TAVILY_API_KEY"), os.getenv("PINECONE_API_KEY")]):
    st.warning("Some API keys are missing from the .env file.")

llm = ChatCohere(
    model="command-r",
    api_key=os.environ.get("COHERE_API_KEY", ""),
    temperature=0.7,
    max_tokens=512
)

tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))

# --- Compliance Config Loader ---
# with open(r"C:\Users\hussa\FastAPI\AGENTIC RAG\Compliance&Workflow\compliance.yaml") as f:
#     cfg = yaml.safe_load(f)
def render(obj):
    """
    Recursively render dicts, lists/tuples, or scalars in a Streamlit-friendly way.
    """
    import streamlit as st

    if isinstance(obj, dict):
        for key, val in obj.items():
            st.markdown(f"**{key.replace('_',' ').title()}:**")
            render(val)

    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj, start=1):
            st.markdown(f"{i}.")
            render(item)

    else:
        # scalar (str, int, etc.)
        st.write(obj)
def load_config(cfg):
    return {
        "rules": cfg["routing_rules"],
        "flows": {
            "civil_risk": cfg["civil_court"]["sop"],
            "high_risk": cfg["high_court"]["sop"]
        },
        "record_cfg": cfg.get("record_lookup")
    }

def build_record_map(record_cfg):
    if not record_cfg:
        return {}, []
    path = record_cfg["file"]
    key_cols = record_cfg["key_column"]
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    m = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        used = [k for k in key_cols if k in reader.fieldnames]
        for row in reader:
            key = tuple(row[k].strip() for k in used)
            m[key].append(row)
    return m, used

# --- Routing Functions ---
def route_static(query, rules):
    for rule in rules:
        pat = rule["match"].get("regex") or rule["match"].get("contains", "")
        if pat and re.search(pat, query, re.IGNORECASE):
            return rule["route_to"], rule["reason"]
    return None, None

def semantic_fallback(query, crime_type, rules):
    rule_texts = []
    for r in rules:
        kws = r["match"].get("contains", [])
        if isinstance(kws, str): kws=[kws]
        if r["match"].get("regex"):
            kws.append(re.sub(r"[.*()\\|]", "", r["match"]["regex"]))
        rule_texts.append(f"{r['route_to']}: {', '.join(kws)}")
    prompt = (
        "Classify the following legal query into 'civil_risk' or 'high_risk'.\n\n" +
        "Regex rules (with keywords):\n" + "\n".join(rule_texts) + "\n\n" +
        f"Query: \"{query}\"\nCase Crime Type: \"{crime_type}\"\n\n" +
        "List all matching keywords in keys_used array. Default to civil_risk. " +
        "Respond with JSON {\"flow\":...,\"reason\":...,\"keys_used\":[...]}."
    )
    resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    try:
        j = json.loads(resp)
        return j.get("flow"), j.get("reason"), j.get("keys_used") or []
    except:
        return None, "Fallback failed; defaulted.", []

# --- Core Handler ---
def handle_request(query, case_id=None, department=None,cfg=r"C:\Users\hussa\FastAPI\AGENTIC RAG\Compliance&Workflow"):
    cfg_data = load_config(cfg)
    rules = cfg_data["rules"]
    flows = cfg_data["flows"]
    record_cfg = cfg_data["record_cfg"]

    static_flow, static_reason = route_static(query, rules)
    records, crime_type = [], ""
    if case_id and record_cfg:
        rec_map, keys = build_record_map(record_cfg)
        tup = (case_id.strip(),) if len(keys)==1 else tuple(case_id)
        records = rec_map.get(tup, [])
        if records:
            crime_type = records[0].get("Crime Type","")
    elif not record_cfg:
        crime_type = "clean record"

    sem_flow, sem_reason, keys_used = semantic_fallback(query, crime_type, rules)
    flow = sem_flow or static_flow or "civil_risk"
    reason = sem_reason or static_reason or "Defaulted to civil_risk"

    return {
        "flow": flow,
        "reason": reason,
        "sop": flows.get(flow, []),
        "records": records,
        "department": department,
        "keys_used": keys_used
    }

# --- Scraping & Chunking Pipelines ---
def search_with_tavily(query):
    return tavily.search(query=query, search_depth="advanced")["results"]

def fetch_block_from_url(url):
    drv = webdriver.Chrome(options=webdriver.ChromeOptions().add_argument("--headless"))
    drv.get(url)
    try:
        elem = WebDriverWait(drv,5).until(
            EC.presence_of_element_located((By.TAG_NAME,"body"))
        )
        text = elem.text
    except:
        text=""
    drv.quit()
    return text

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0,len(words),max_words)]

def tavily_selenium_pipeline(query, chunk_size=150):
    res = search_with_tavily(query)
    if not res: return {"chunk_1":"No results"}
    url = res[0]["url"]
    txt = fetch_block_from_url(url)
    clean = re.sub(r'\s+',' ',txt)
    chunks = chunk_text(clean,chunk_size)
    return {f"chunk_{i+1}":c for i,c in enumerate(chunks)}

def document_query_pipeline(file_path, query):
    text = open(file_path,encoding="utf-8").read()
    chunks = chunk_text(text,200)
    matched = [c for c in chunks if all(t.lower() in c.lower() for t in re.findall(r"\w+",query))]
    if not matched: return {"chunk_1":"No matching context"}
    return {f"chunk_{i+1}":c for i,c in enumerate(matched)}
@tool("web_search")
def web_search(qry_str: str) -> str:
    """
    Search hklii.hk via Tavily + Selenium.
    Input: a keyword string (qry_str).
    Output: a JSON‐encoded dict of text chunks.
    """
    return json.dumps(tavily_selenium_pipeline(qry_str, chunk_size=150),ensure_ascii=False)

@tool("doc_search")
def doc_search(file_path: str, qry_str: str) -> str:
    """
    Search a local document via n-grams.
    Input:
      - file_path: path to the document
      - qry_str:   cleaned keyword string
    Output: a JSON‐encoded dict of matching text chunks.
    """
    return json.dumps(document_query_pipeline(file_path, qry_str),ensure_ascii=False)

@tool("fetch_practical_law")
def fetch_practical_law(qry: str) -> str:
    """
    Search Practical Law – Hong Kong (Thomson Reuters) via Tavily + Selenium.
    Input: a keyword string (qry).
    Output: a JSON‐encoded dict of text chunks.
    """
    return json.dumps(tavily_selenium_pipeline(qry, chunk_size=150),ensure_ascii=False)

@tool("fetch_hkel")
def fetch_hkel(qry: str) -> str:
    """
    Search HKeL via Tavily + Selenium.
    Input: a keyword string (qry).
    Output: a JSON‐encoded dict of text chunks.
    """
    return json.dumps(tavily_selenium_pipeline(qry, chunk_size=150),ensure_ascii=False)

@tool("fetch_halsbury")
def fetch_halsbury(qry: str) -> str:
    """
    Search Practical Law HK, and Halsbury’s HK via Tavily + Selenium.
    Input: a keyword string (qry).
    Output: a JSON‐encoded dict of text chunks.
    """
    return json.dumps(tavily_selenium_pipeline(qry, chunk_size=150), ensure_ascii=False)

@tool("cross_verifier")
def cross_verifier(qry_json: str) -> str:
    """
    Input: JSON-encoded dict:
        {"HKeL": [...], "PracticalLaw": [...], "Halsbury": [...]}
    Output: JSON-encoded cross-verified summary.
    """
    data = json.loads(qry_json)
    # implement cross-verification logic here
    return json.dumps({"summary": "Cross-verified output"}, ensure_ascii=False)

# --- Agents ---
agent_web = Agent(
    role="Web Searcher",
    goal="Find relevant legal information from hklii.hk based on {qry_str} and {department}.",
    backstory="Fetch SOPs and forms via web_search.",
    tools=[web_search],
    llm=llm,
    verbose=True
)
agent_doc = Agent(
    role="Doc & Conditional Web Searcher",
    goal="Extract SOPs from local docs (points 1–8), then list Serve documents via web (point 9) all this should be implemented on the basis of context from {qry_str}.",
    backstory=(
        "First use `doc_search` to handle points 1–8. "
        "Only for point 9, call `web_search` to fetch Serve documents from hklii.hk."
    ),
    tools=[doc_search, web_search],
    llm=llm,
    verbose=True
)
agent_deep = Agent(
   role="Hong Kong Legal Deep Researcher",
   goal=(
        "1. Fetch each reference from HKeL, Practical Law – Hong Kong (Thomson Reuters), Practical Law HK, and Halsbury’s HK; "
        "2. Cross-verify overlapping text and commentary; "
        "3. Synthesize a unified summary; "
        "4. Validate whether the resulting analysis pertains to civil or High Court jurisdiction."
    ),
   backstory=(
        "You’re an expert in HK legislation and case law. "
        "You must ensure that no discrepancies slip through when triangulating between statutory text, practitioner commentary, and general legal treatise."
    ),
    tools=[fetch_hkel, fetch_practical_law, fetch_halsbury, cross_verifier],
    llm=llm,
    verbose=True
)

# --- Tasks ---
task_web = Task(
        description=("Department: {department}\nQuery: {qry_str}\n"
                    "Only use web_search tool to get the information"
        ),
        expected_output=(
            "1. Provide a title of the case and articles in smaller prompt as references if there is any.\n"
            "2. Give a detailed answer like it is a case study in a decision tree format.\n"
            "3. Correct the mistakes in SOPs provided or anything that was missed should be mentioned and explain the necessity of that point.\n"
            "4. try to make the answer as authentic as possible by backing the statement with references.\n"
            "5. List the sources or citations used and their respective links.\n"
            "6. Include a sub-heading that indicates the tool used to obtain the source ('web_search').\n"
            "7. Bring the following refrences , Articles and Forms mentioned to back the statemnt.\n"
            "8. If they use documents as a reference then return the name of those documents with its extension.\n"
            "9. search that document and its contents.\n"
            "10. return the references and respective information under the heading of those documents.\n"
            "11. List all the required Serve documents from the applicant or defendants side via web search on the basis of the charge on defendant.\n"
            "12. Mention the name of the Governing body (Department) if provided in the {department} field.\n"
            "13. If not provided then choose one according to the charges and analyzing the current status of the case.\n"
            "14. Also mention the current Crime Convict has commited and is on trial for it can be multiple charges or it can be one but there should be a heading of it.\n"
            "If SOPs are not provided then use web_search to get updated SOPs for that particular case on the basis of his or her past records, current charges and the department that is handling the case.\n"
        ),
        agent=agent_web
    )
task_doc = Task(
    description=(
        "Department: {department}\nQuery: {qry_str}\n"
        "1-8: Use only doc_search; 9-14: Use web_search"
    ),
    expected_output=(
            "1–8: **Use only** the `doc_search` tool to find and quote chunks.\n"
            "  1. Provide a title of the case and articles…\n"
            "  2. Give a detailed answer like it is a case study in a decision tree format.\n"
            "  3. Correct the mistakes in SOPs provided or anything that was missed should be mentioned and explain the necessity of that point.\n"
            "  4. Back statements with references…\n"
            "  5. List sources and links…\n"
            "  6. Sub-heading: tool used (`doc_search`).\n"
            "  7. Bring referenced Articles and Forms…\n"
            "  8. If documents referenced, return their filenames…\n"
            "  9. Search that document’s contents and return under headings.\n\n"
            "10: **Use only** the `web_search` tool to list all required Serve documents "
            "from the applicant’s or defendant’s side, based on the defendant’s charge.\n"
            "11. Mention the name of the Governing body (Department) if provided in the {department} field.\n"
            "12. If not provided then choose one according to the charges and analyzing the current status of the case on `web_search` tool.\n"
            "13. Also mention the current Crime Convict has commited and is on trial for it can be multiple charges or it can be one but there should be a heading of it.\n"
            "14. If SOPs are not provided then use web_search to get updated SOPs for that particular case on the basis of his or her past records, current charges and the department that is handling the case.\n"
        ),
   agent=agent_doc
)
task_deep = Task(
    description=(
        "Department: {department}\nQuery: {qry_str}\n"
        "Cross-verify and synthesize summary."
    ),
    expected_output=(
        "  1. Use only the provided articles and forms from the previous responses.\n"
        "  2. search each articles and forms separately using all three of the web searching tools (fetch_hkel, fetch_practical_law, fetch_halsbury).\n"
        "  3. Cross-verify overlapping text and commentary.\n"
        "  4. Synthesize a unified summary.\n"
        "  5. Validate whether the resulting analysis pertains to civil or High Court jurisdiction.\n"
        "  6. Explain the information for those articles and forms that has been received from previous agents output in 'qry' or 'resp' field.\n"
        "  7. List all sources the Deep Search Agent referenced, including tool names and any document identifiers if there are any.\n"
        "  8. Explain all the legal terms in simple english.\n"
        ),
  agent=agent_deep
)

cfg_folder=Path(__file__).parent
# --- Streamlit UI ---
st.set_page_config(page_title="Legal Compliance Chatbot")
st.title("Legal Compliance Chatbot")

# Sidebar: mode & file selection
mode = st.sidebar.radio("Mode", ["Web Only", "With Document"])
doc_file = None
if mode == "With Document":
    csvs = list(cfg_folder.glob("*.csv"))
    sel = st.sidebar.selectbox("Select record CSV", [p.name for p in csvs])
    doc_file = str(cfg_folder / sel)

# Sidebar: compliance YAML
yamls = list(cfg_folder.glob("*.yaml"))
sel_yaml = st.sidebar.selectbox("Select compliance YAML", [p.name for p in yamls])
cfg = yaml.safe_load((cfg_folder / sel_yaml).open(encoding="utf-8"))

# User inputs
query = st.text_input("Enter legal query:")
case_id = st.text_input("Case ID (optional):")
department = st.text_input("Department (optional):")

if st.button("Submit"):
    if not query:
        st.error("Please enter a query.")
    else:
        res = handle_request(query, case_id or None, department or None, cfg)
        qry_str = f"flow:{res['flow']}\ncrime type:{', '.join(res['keys_used'])}\nlocation:{department or 'Unknown'}"
        inputs = {"qry_str": qry_str, "department": department or None}
        tasks = [task_web, task_deep] if mode == "Web Only" else [task_doc, task_deep]
        crew = Crew(
            agents=[agent_doc, agent_web, agent_deep],
            tasks=tasks,
            llm=llm,
            verbose=True
        )       
        output = crew.kickoff(inputs)
        st.subheader("Routing Decision")
        st.markdown(f"**Flow:** {res['flow']}")

        st.subheader("Standard Operating Procedure (SOP)")
        if res['sop']:
            for i, step in enumerate(res['sop'], 1):
                st.markdown(f"{i}. {step}")
        else:
            st.markdown("_No SOP steps defined for this flow._")

        if res['records']:
            with st.expander(f"🔍 {len(res['records'])} Record(s) Found"):
                import pandas as pd
                df = pd.DataFrame(res['records'])
                st.dataframe(df)

        if res.get('department'):
            st.markdown(f"**Department:** {res['department']}")
        if res.get('keys_used'):
            st.markdown(f"**Keys Used:** {', '.join(res['keys_used'])}")

        st.subheader("Agent Responses")
        out_dict = output.dict()  # or: json.loads(output.json())

        if out_dict:
            # Grab the first agent name & its result
            first_agent_name, first_out = next(iter(out_dict.items()))

            with st.expander(f"{first_agent_name} Output", expanded=True):
                render(first_out)
        else:
            st.warning("No agent output available.")
