# 1_fetch_papers.py
import arxiv
import pandas as pd
import time

# --- Configuration ---
QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:cs.NE"
MAX_RESULTS = 500

print(f"[*] Starting search for {MAX_RESULTS} papers with query: '{QUERY}'...")

# --- Configure the Client ---
# This is the key to avoiding rate-limit errors.
# It tells the client to fetch 100 results per request and wait 3 seconds between requests.
client = arxiv.Client(
  page_size = 100,
  delay_seconds = 3,
  num_retries = 5
)

# Define the search object
search = arxiv.Search(
  query=QUERY,
  max_results=MAX_RESULTS,
  sort_by=arxiv.SortCriterion.SubmittedDate
)

# --- Process Results ---
# Use the modern syntax: client.results(search)
# This also uses the client configuration we set above.
results_generator = client.results(search)

papers_data = []
try:
    for i, result in enumerate(results_generator):
        # Print progress every 25 papers
        if (i + 1) % 25 == 0:
            print(f"[*] Fetched {i + 1} / {MAX_RESULTS} papers...")

        paper_data = {
            'id': result.get_short_id(),
            'title': result.title,
            'abstract': result.summary.replace('\n', ' '),
            'authors': [author.name for author in result.authors],
            'date': result.published.strftime('%Y-%m-%d')
        }
        papers_data.append(paper_data)

# This error can happen if the API is unstable. We'll catch it gracefully.
except arxiv.UnexpectedEmptyPageError as e:
    print(f"\n[!] Error: Encountered an empty page from arXiv API. This can happen under heavy load.")
    print(f"[!] Halting search, but will save the {len(papers_data)} papers already downloaded.")
    print(f"[!] Details: {e}")

finally:
    # This block will run whether there was an error or not
    if not papers_data:
        print("\n[❌] No papers were downloaded. Exiting.")
    else:
        print(f"\n[*] Successfully fetched {len(papers_data)} papers.")
        # Create a pandas DataFrame
        df = pd.DataFrame(papers_data)
        # Export the DataFrame to a CSV file
        df.to_csv('papers_metadata.csv', index=False)
        print(f"[✅] Complete! Data saved to papers_metadata.csv")