import sys

from src.topic_search import fetch_topic_literature
from src.database import init_db, insert_papers


def main():
    topic = input("Research topic: ").strip()
    if not topic:
        print("Please enter a topic.")
        return

    try:
        start_year = int(input("Start year (e.g. 2020): ").strip())
        end_year = int(input("End year (e.g. 2026): ").strip())
        max_results = int(input("Maximum papers to request (300-500): ").strip())
    except ValueError:
        print("Years and paper limit must be whole numbers.")
        return

    if start_year < 2000 or end_year > 2026 or start_year > end_year:
        print("Use years from 2000 to 2026, with start year <= end year.")
        return

    if not 300 <= max_results <= 500:
        print("Choose a paper limit between 300 and 500.")
        return

    print(f"\nSearching PubMed for: {topic}")
    print(f"Years: {start_year}-{end_year}")
    print(f"Requesting up to {max_results} papers...\n")

    try:
        articles = fetch_topic_literature(
            topic, start_year, end_year, max_results=max_results
        )
    except Exception as exc:
        print(f"PubMed retrieval failed: {exc}")
        sys.exit(1)

    print(f"Retrieved {len(articles)} papers with usable abstracts.")

    if not articles:
        print("Nothing to save.")
        return

    init_db()
    inserted = insert_papers(articles, query_topic=topic)

    print(f"New papers saved: {inserted}")
    print(f"Already stored for this topic: {len(articles) - inserted}")
    print("Retrieval complete. No LLM generation was run.")


if __name__ == "__main__":
    main()
