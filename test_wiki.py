import wikipedia

# Define species mapping
species_mapping = {
    "turmeric_tree": "Berberis_aristata",
    # Add other species as needed, e.g.:
    # "neem_tree": "Azadirachta indica",
    # "mango": "Mangifera indica"
}

# Test with a specific species
species = "turmeric_tree"
wiki_title = species_mapping.get(species, species)
try:
    page = wikipedia.page(wiki_title)
    print(f"Title: {page.title}")
    print(f"Summary: {wikipedia.summary(wiki_title, sentences=3)}")
except wikipedia.exceptions.PageError:
    print(f"No Wikipedia page found for '{wiki_title}'.")
except Exception as e:
    print(f"Error fetching Wikipedia info: {e}")