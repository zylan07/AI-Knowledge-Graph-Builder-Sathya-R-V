from neo4j import GraphDatabase
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class Job:
    job_id: str
    category: str
    workplace: str
    employment_type: str
    priority_class: str
    demand_score: float
    city: str
    country: str
    region: str
    department: str
    department_category: str
    is_active: bool
    text_description: str

def load_jobs_from_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    MATCH (j:Job)-[:LOCATED_IN]->(l:Location),
          (j)-[:IN_DEPARTMENT]->(d:Department),
          (j)-[:BELONGS_TO]->(c:Category)
    OPTIONAL MATCH (j)-[:REQUIRES]->(s:Skill)
    RETURN
        j.id AS job_id,
        c.name AS category,
        j.workplace AS workplace,
        j.employment_type AS employment_type,
        j.priority_class AS priority_class,
        toFloat(j.demand_score) AS demand_score,
        l.city AS city,
        l.country AS country,
        l.region AS region,
        d.name AS department,
        d.category AS department_category,
        j.is_active AS is_active,
        collect(DISTINCT s.name) AS skills
    """
    jobs = []
    with driver.session() as session:
        for record in session.run(query):
            skills_list = record["skills"] or []
            text = (
                f"Job: {record['category']}\n"
                f"Location: {record['city']}, {record['country']} ({record['region']} region)\n"
                f"Work: {record['workplace']} {record['employment_type']}\n"
                f"Department: {record['department']} ({record['department_category']})\n"
                f"Priority: {record['priority_class']}\n"
                f"Demand Score: {record['demand_score']:.1f}/100\n"
                f"Required Skills: {', '.join(skills_list) if skills_list else 'Not specified'}"
            )
            jobs.append(Job(
                job_id=record['job_id'],
                category=record['category'],
                workplace=record['workplace'],
                employment_type=record['employment_type'],
                priority_class=record['priority_class'],
                demand_score=float(record['demand_score']) if record['demand_score'] else 0.0,
                city=record['city'] or 'Unknown',
                country=record['country'] or 'Unknown',
                region=record['region'] or 'Unknown',
                department=record['department'] or 'Unknown',
                department_category=record['department_category'] or 'Unknown',
                is_active=bool(record['is_active']),
                text_description=text
            ))
    driver.close()
    return jobs

def load_graph_data(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes, edges = [], []
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN elementId(n) AS eid, labels(n)[0] AS label,
                       coalesce(n.id, n.name, elementId(n)) AS display_id
                LIMIT 1300
            """)
            for r in result:
                eid = str(r["eid"])
                nodes.append({"id": eid, "label": r["label"], "name": str(r["display_id"])})
        with driver.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN elementId(a) AS src, elementId(b) AS tgt, type(r) AS rel
                LIMIT 5243
            """)
            for r in result:
                edges.append({"src": str(r["src"]), "tgt": str(r["tgt"]), "rel": r["rel"]})
    finally:
        driver.close()
    return nodes, edges

def load_stats(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    stats = {}
    try:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt")
            stats["nodes"] = {r["label"]: r["cnt"] for r in result}
        with driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt")
            stats["edges"] = {r["rel"]: r["cnt"] for r in result}
        with driver.session() as session:
            result = session.run("MATCH (j:Job)-[:REQUIRES]->(s:Skill) RETURN s.name AS skill, count(j) AS cnt ORDER BY cnt DESC LIMIT 20")
            stats["top_skills"] = [(r["skill"], r["cnt"]) for r in result]
    finally:
        driver.close()
    return stats


def get_node_details_from_neo4j(uri, user, password, node_name, node_label):
    """Fetch full details of a specific node from Neo4j for the AI Agent"""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    details = {"name": node_name, "label": node_label, "properties": {}, "relationships": []}
    try:
        with driver.session() as session:
            if node_label == "Job":
                result = session.run("""
                    MATCH (j:Job {id: $name})
                    OPTIONAL MATCH (j)-[:LOCATED_IN]->(l:Location)
                    OPTIONAL MATCH (j)-[:IN_DEPARTMENT]->(d:Department)
                    OPTIONAL MATCH (j)-[:BELONGS_TO]->(c:Category)
                    OPTIONAL MATCH (j)-[:REQUIRES]->(s:Skill)
                    RETURN j, l, d, c, collect(DISTINCT s.name) AS skills
                    LIMIT 1
                """, name=node_name)
                for r in result:
                    details["properties"] = dict(r["j"])
                    rels = []
                    if r["l"]: rels.append(f"LOCATED_IN -> {r['l'].get('city','?')}, {r['l'].get('country','?')}")
                    if r["d"]: rels.append(f"IN_DEPARTMENT -> {r['d'].get('name','?')}")
                    if r["c"]: rels.append(f"BELONGS_TO -> {r['c'].get('name','?')}")
                    rels.append(f"REQUIRES -> {', '.join(r['skills']) if r['skills'] else 'None'}")
                    details["relationships"] = rels
            elif node_label == "Skill":
                result = session.run("""
                    MATCH (s:Skill {name: $name})
                    OPTIONAL MATCH (j:Job)-[:REQUIRES]->(s)
                    RETURN s, count(j) AS job_count, collect(DISTINCT j.id)[0..5] AS sample_jobs
                """, name=node_name)
                for r in result:
                    details["properties"] = {"name": node_name, "job_count": r["job_count"]}
                    details["relationships"] = [f"REQUIRED_BY {r['job_count']} jobs", f"Sample: {', '.join(r['sample_jobs'])}"]
            elif node_label == "Location":
                result = session.run("""
                    MATCH (l:Location {city: $name})
                    OPTIONAL MATCH (j:Job)-[:LOCATED_IN]->(l)
                    RETURN l, count(j) AS job_count
                """, name=node_name)
                for r in result:
                    details["properties"] = dict(r["l"])
                    details["relationships"] = [f"HAS {r['job_count']} jobs located here"]
            elif node_label == "Department":
                result = session.run("""
                    MATCH (d:Department {name: $name})
                    OPTIONAL MATCH (j:Job)-[:IN_DEPARTMENT]->(d)
                    RETURN d, count(j) AS job_count
                """, name=node_name)
                for r in result:
                    details["properties"] = dict(r["d"])
                    details["relationships"] = [f"HAS {r['job_count']} jobs in this department"]
            elif node_label == "Category":
                result = session.run("""
                    MATCH (c:Category {name: $name})
                    OPTIONAL MATCH (j:Job)-[:BELONGS_TO]->(c)
                    RETURN c, count(j) AS job_count
                """, name=node_name)
                for r in result:
                    details["properties"] = dict(r["c"])
                    details["relationships"] = [f"HAS {r['job_count']} jobs in this category"]
    except Exception as e:
        details["error"] = str(e)
    finally:
        driver.close()
    return details

print("graph_utils.py written!")
