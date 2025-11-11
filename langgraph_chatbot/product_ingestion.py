"""
FAISS Vector Store Ingestion Script for ZUS Coffee Products

This script:
1. Loads product data from ZUS_PRODUCTS.js file
2. Generates embeddings using sentence-transformers
3. Creates FAISS index for vector similarity search
4. Saves index and metadata to disk

Run with: python ingest_products.py
"""

import json
import pickle
import re
import numpy as np
from pathlib import Path
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os


# Paths
DATA_DIR = Path("data")
# FAISS_INDEX_PATH = DATA_DIR / "products.index"
# METADATA_PATH = DATA_DIR / "products_metadata.pkl"
# PRODUCTS_JS_PATH = "ZUS_PRODUCTS.js"

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

load_dotenv()
product_raw = os.getenv('PRODUCTS_RAW_PATH')
product_index = os.getenv('PRODUCT_INDEX_PATH')
product_metadata = os.getenv('PRODUCT_METADATA_PATH')


# LOAD PRODUCTS FROM JAVASCRIPT FILE
def strip_html_tags(html_text: str) -> str:
    """Remove HTML tags from text"""
    if not html_text:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', html_text).strip()


def load_products_from_js(file_path: str) -> List[Dict]:
    """
    Load products from JavaScript file.
    Extracts the PRODUCTS_DATA array from the JS file.
    """
    print(f"Loading products from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the PRODUCTS_DATA array
        # Extract JSON array between [ and ];
        match = re.search(r'(?:const|var|let)?\s*PRODUCTS_DATA\s*=\s*(\[[\s\S]*\]);?', content)
        
        if not match:
            raise ValueError("Could not find PRODUCTS_DATA in JavaScript file")
        
        json_str = match.group(1)
        products = json.loads(json_str)
        
        print(f"✓ Loaded {len(products)} products from JavaScript file")
        return products
        
    except FileNotFoundError:
        print(f"✗ Error: {file_path} not found!")
        raise
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing JSON from JavaScript file: {e}")
        raise
    except Exception as e:
        print(f"✗ Error loading products: {e}")
        raise


def transform_product_for_search(product: Dict) -> Dict:
    """
    Transform Shopify product format to search-optimized format.
    Extracts key information and creates searchable fields.
    """
    # Clean HTML from body
    description = strip_html_tags(product.get('body_html', ''))
    
    # Extract variant information
    variants = product.get('variants', [])
    variant_titles = [v.get('title', '') for v in variants if v.get('title') != 'Default Title']
    prices = [float(v.get('price', 0)) for v in variants if v.get('price')]
    
    # Get primary price (lowest variant price)
    price = min(prices) if prices else 0.0
    
    # Extract tags
    tags = product.get('tags', '')
    tag_list = [t.strip() for t in tags.split(',') if t.strip()] if isinstance(tags, str) else []
    
    # Build features list
    features = []
    features.extend(tag_list)
    features.extend(variant_titles)
    features.append(product.get('product_type', ''))
    features.append(product.get('vendor', ''))
    
    # Remove empty and duplicates
    features = list(set([f for f in features if f]))
    
    return {
        'id': str(product.get('id', '')),
        'title': product.get('title', ''),
        'description': description,
        'price': price,
        'category': product.get('product_type', 'General'),
        'vendor': product.get('vendor', ''),
        'tags': tag_list,
        'variants': variant_titles,
        'features': features,
        'handle': product.get('handle', '')
    }


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

def create_product_text(product: Dict) -> str:
    """
    Create searchable text from product data.
    This combines all relevant fields for embedding.
    """
    text_parts = [
        f"Product: {product['title']}",
        f"Category: {product['category']}",
        f"Description: {product['description']}"
    ]
    
    if product.get('variants'):
        text_parts.append(f"Variants: {', '.join(product['variants'])}")
    
    if product.get('features'):
        text_parts.append(f"Features: {', '.join(product['features'])}")
    
    if product.get('tags'):
        text_parts.append(f"Tags: {', '.join(product['tags'])}")
    
    text_parts.append(f"Price: RM{product['price']:.2f}")
    
    return " | ".join(text_parts)


def generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    Returns numpy array of shape (n_texts, embedding_dim)
    """
    print(f"Generating embeddings for {len(texts)} products...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    print('em', embeddings)
    return embeddings


# ============================================================================
# FAISS INDEX CREATION
# ============================================================================

def create_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """
    Create FAISS index from embeddings.
    
    Using IndexFlatIP (Inner Product) with normalized vectors = cosine similarity
    For larger datasets, consider IndexIVFFlat or IndexHNSW
    """
    dimension = embeddings.shape[1]
    
    print(f"Creating FAISS index (dimension={dimension})...")
    
    # IndexFlatIP: Exact search using inner product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    
    # For larger datasets (>10k products), use this instead:
    # nlist = 100  # number of clusters
    # quantizer = faiss.IndexFlatIP(dimension)
    # index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    # index.train(embeddings)
    
    # Add vectors to index
    index.add(embeddings)
    print('index', index)
    print(f"✓ Index created with {index.ntotal} vectors")
    
    return index


# ============================================================================
# MAIN INGESTION PIPELINE
# ============================================================================

def ingest_products():
    """Main ingestion pipeline"""
    
    print("\n" + "="*70)
    print("FAISS PRODUCT INGESTION PIPELINE - ZUS COFFEE")
    print("="*70 + "\n")
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Step 1: Load products from JavaScript file
    print("Step 1: Loading products from JavaScript file...")
    raw_products = load_products_from_js(product_raw)
    print()
    
    # Step 2: Transform products for search
    print("Step 2: Transforming products for search...")
    products = [transform_product_for_search(p) for p in raw_products]
    print(f"✓ Transformed {len(products)} products\n")
    
    # Save transformed products to JSON for reference
    products_json_path = DATA_DIR / "products.json"
    with open(products_json_path, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved transformed products to {products_json_path}\n")
    
    # Step 3: Create searchable text for each product
    print("Step 3: Creating searchable text...")
    product_texts = [create_product_text(p) for p in products]
    print(product_texts)
    print(f"✓ Created text for {len(product_texts)} products\n")
    
    # Step 4: Load embedding model
    print(f"Step 4: Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Model loaded (dimension={model.get_sentence_embedding_dimension()})\n")
    
    # Step 5: Generate embeddings
    print("Step 5: Generating embeddings...")
    embeddings = generate_embeddings(product_texts, model)
    print(f"✓ Generated embeddings with shape {embeddings.shape}\n")
    
    # Step 6: Create FAISS index
    print("Step 6: Creating FAISS index...")
    index = create_faiss_index(embeddings)
    print()
    
    # Step 7: Save index and metadata
    print("Step 7: Saving to disk...")
    
    # Save FAISS index
    faiss.write_index(index, str(product_index))
    print(f"✓ Saved FAISS index to {product_index}")
    
    # Save metadata (product info without embeddings)
    metadata = {
        'products': products,
        'model_name': EMBEDDING_MODEL,
        'embedding_dimension': EMBEDDING_DIMENSION,
        'index_size': index.ntotal
    }
    
    with open(product_metadata, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata to {product_metadata}\n")
    
    # Summary
    print("="*70)
    print("INGESTION COMPLETE")
    print("="*70)
    print(f"Products indexed: {len(products)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Index file: {product_index}")
    print(f"Metadata file: {product_metadata}")
    print("\nProducts by category:")
    categories = {}
    for p in products:
        cat = p['category']
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")
    print("="*70 + "\n")


# ============================================================================
# UTILITY: TEST SEARCH
# ============================================================================

def test_search(query: str, top_k: int = 3):
    """Test the created index with a sample query"""
    
    print(f"\nTesting search for: '{query}'")
    print("-" * 50)
    
    # Load model
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Load index
    index = faiss.read_index(str(product_index))
    
    # Load metadata
    with open(product_metadata, 'rb') as f:
        metadata = pickle.load(f)
    
    products = metadata['products']
    
    # Generate query embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Display results
    print(f"\nTop {top_k} results:\n")
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
        product = products[idx]
        print(f"{i}. {product['title']} (score: {distance:.4f})")
        # print(f"   Category: {product['category']}")
        # print(f"   Price: RM {product['price']:.2f}")
        # if product.get('variants'):
        #     print(f"   Variants: {', '.join(product['variants'][:3])}")
        # print(f"   Description: {product['description'][:100]}...")
        print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Run ingestion
    ingest_products()
    
    # Run test searches
    print("\n" + "="*70)
    print("TESTING SEARCHES")
    print("="*70)
    
    test_queries = [
        "any zus coffee recommend?",
        "stainless steel straw reusable",
        "malaysian heritage cup sleeve",
        "tumbler all day cup",
        "bundle set corak malaysia"
    ]
    
    for query in test_queries:
        test_search(query, top_k=3)
    
    print("="*70)
    print("All done! Use the FAISS index in your FastAPI server.")
    print("="*70 + "\n")