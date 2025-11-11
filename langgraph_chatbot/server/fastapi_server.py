from fastapi import FastAPI, HTTPException, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, validator
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pathlib import Path
from datetime import datetime
import sqlite3
import pickle
import re
import faiss
import os
import html
import asyncio


load_dotenv()
api_key_openai = os.getenv("OPENAI_API_KEY")
product_index = Path(os.getenv('PRODUCT_INDEX_PATH'))
product_metadata = Path(os.getenv('PRODUCT_METADATA_PATH'))

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMULATE_DOWNTIME = False
SIMULATE_SLOW_API = False
MAX_REQUEST_SIZE = 10000  # 10KB max request size
MAX_QUERY_LENGTH = 500  # Max query string length

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=api_key_openai)
    use_llm = True
except Exception as e:
    print(f"  - Warning: Could not initialize OpenAI client: {e}")
    use_llm = False


def sanitize_input(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """
    Sanitize user input to prevent XSS and other attacks.

    Args:
        text: User input string
        max_length: Maximum allowed length

    Returns:
        Sanitized string

    Raises:
        ValueError: If input is too long or contains dangerous patterns
    """
    if not text:
        raise ValueError("Input cannot be empty")

    # Check length
    if len(text) > max_length:
        raise ValueError(f"Input too long (max {max_length} characters)")

    # Check for suspicious patterns BEFORE escaping
    dangerous_patterns = [
        r'<script',
        r'</script>',
        r'javascript:',
        r'on\w+\s*=',  # Event handlers like onclick=
        r'eval\(',
        r'exec\(',
        r'<iframe',
        r'<object',
        r'<embed',
    ]

    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, text_lower):
            raise ValueError("Input contains potentially dangerous content")

    # HTML escape to prevent XSS (after validation)
    text = html.escape(text)

    return text.strip()

def validate_sql_injection(query: str) -> None:
    """
    Enhanced SQL injection detection.

    Args:
        query: Natural language query

    Raises:
        HTTPException: If SQL injection attempt detected
    """
    query_lower = query.lower()

    # Check for SQL injection patterns
    sql_injection_patterns = [
        re.compile(r"\b(drop|delete|update|insert|alter|create|truncate|replace)\b", re.IGNORECASE),
        re.compile(r"\bunion\b", re.IGNORECASE),
        re.compile(r"\bselect\b\s+.*\bfrom\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"--|\b--\b"),
        re.compile(r"/\*.*\*/", re.DOTALL),
        re.compile(r"\bexec\b|\bexecute\b", re.IGNORECASE),
        re.compile(r"\bdeclare\b\s+@", re.IGNORECASE),
        re.compile(r"\bcast\(|\bconvert\(|\bchar\(|\bnchar\(|\bvarchar\(|\bnvarchar\(", re.IGNORECASE),
        re.compile(r"0x[0-9a-f]+", re.IGNORECASE),
        re.compile(r"['\"].*\b(or|and)\b.*['\"]", re.IGNORECASE),
        re.compile(r"\bor\s+\d+\s*=\s*\d+\b", re.IGNORECASE), 
    ]

    for pattern in sql_injection_patterns:
        if re.search(pattern, query_lower):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "malicious_input_detected",
                    "message": "Your query contains potentially malicious content. Please rephrase your question naturally.",
                    "recovery_hint": "Ask your question without special SQL characters or commands."
                }
            )



# GLOBAL STATE (Loaded on startup)
class VectorStore:
    """Singleton for FAISS index and embedding model"""
    
    def __init__(self):
        self.index = None     
        self.model = None
        self.products = None
        self.metadata = None
        self.loaded = False
    
    def load(self):
        """Load FAISS index and model"""
        
        if self.loaded:
            return
        
        try:
            # Check if files exist
            if not product_index.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {product_index}. "
                    "Run 'python ingest_products.py' first."
                )
            
            if not product_metadata.exists():
                raise FileNotFoundError(
                    f"Metadata not found at {product_metadata}. "
                    "Run 'python ingest_products.py' first."
                )
            
            # Load FAISS index
            self.index = faiss.read_index(str(product_index))
            print(f"\n   - Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load metadata
            with open(product_metadata, 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.products = self.metadata['products']
 
            # Load embedding model
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            
            self.loaded = True
            
        except Exception as e:
            print(f"✗ Error loading vector store: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded products
        
        Used in /health
        """
        if not self.loaded:
            return {"error": "Vector store not loaded"}
        
        categories = {}
        total_variants = 0
        price_range = {"min": float('inf'), "max": 0}
        
        for p in self.products:
            # Category count
            cat = p.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Variant count
            total_variants += len(p.get('variants', []))
            
            # Price range
            price = p.get('price', 0)
            if price > 0:
                price_range["min"] = min(price_range["min"], price)
                price_range["max"] = max(price_range["max"], price)
        
        return {
            "total_products": len(self.products),
            "total_variants": total_variants,
            "categories": categories,
            "price_range": price_range if price_range["min"] != float('inf') else None,
            "embedding_model": self.metadata.get('model_name'),
            "embedding_dimension": self.metadata.get('embedding_dimension')
        }

# Initialize vector store
vector_store = VectorStore()


# Initialize database for outlets on startup
def init_outlets_db():
    """Initialize SQLite database with outlets data"""
    
    conn = sqlite3.connect('outlets.db')
    cursor = conn.cursor()
    
    # Create outlets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS outlets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            address TEXT NOT NULL,
            opening_time TEXT NOT NULL,
            closing_time TEXT NOT NULL,
            phone TEXT
        )
    ''')
    
    # Check if data already exists
    cursor.execute('SELECT COUNT(*) FROM outlets')
    if cursor.fetchone()[0] == 0:
        # Insert sample data
        outlets_data = [
            ("SS2", "Petaling Jaya", "5, Jalan SS 2/67, SS 2, 47300 Petaling Jaya", "07:00:00", "21:40:00", "012-8161340"),
            ("Uptown Damansara", "Petaling Jaya", "44-G (Ground Floor, Jalan SS21/39, Damansara Utama, 47400 Petaling Jaya", "07:00:00", "22:40:00", "012-8161340"),
            ("Suria KLCC", "Kuala Lumpur", "Lot No. OS301 , Level 3, Petronas Twin Tower, Persiaran Petronas, Kuala Lumpur City Centre, 50088 Kuala Lumpur", "08:00:00", "21:40:00", "012-8161340"),
            ("Bangsar South", "Kuala Lumpur", "Phase 2, No.8, DKLS Tower, Ground Floor, Tower 8, Avenue 5, The Horizon, Jalan Kerinchi, Bangsar South, 59200 Kuala Lumpur", "07:00:00", "19:40:00", "012-8161340"),
            ("Mid Valley", "Kuala Lumpur", "SECOND FLOOR, Lot SK-10, Lingkaran Syed Putra, Mid Valley City, 59200 Kuala Lumpur", "08:00:00", "21:40:00", "012-8161340"),
            ("SS15", "Subang Jaya", "No.106-G (Ground Floor, Jalan SS 15/4b, Ss 15, 47500 Subang Jaya", "07:00:00", "22:40:00", "012-8161340"),
        ]
        
        cursor.executemany('''
            INSERT INTO outlets (name, location, address, opening_time, closing_time, phone)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', outlets_data)
        
        conn.commit()
        print(f" - Initialized outlets database with {len(outlets_data)} outlets")
    
    conn.close()


# LIFESPAN CONTEXT MANAGER - Initial Loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    
    # Startup
    # Initialize outlet database
    try:
        init_outlets_db()
    except Exception as e:
        print(f"   - Warning: Could not initialize outlets database: {e}")

    # Load vector store
    try:
        vector_store.load()
        print("   - API ready with FAISS vector search & SQLite Database\n")
    except Exception as e:
        print(f"  - Failed to load vector store: {e}")
        print("Run 'python ingest_products.py' to create FAISS index\n")

    yield



# FASTAPI APP INITIALIZATION
app = FastAPI(
    title="ZUS Coffee Custom API",
    description="Simple calculator API for arithmetic operation, FAISS RAG for products and Text2SQL for outlets",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MIDDLEWARE
@app.middleware("http")
async def validate_request_size(request: Request, call_next):
    """Middleware to check request size"""
    if request.method in ["POST", "PUT", "PATCH"]:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large",
                    "message": f"Request body too large (max {MAX_REQUEST_SIZE} bytes)",
                    "recovery_hint": "Please reduce the size of your request"
                }
            )
    
    response = await call_next(request)
    return response


@app.middleware("http")
async def simulate_failures(request: Request, call_next):
    """Middleware to simulate API failures for testing"""
    
    # Simulate slow API
    if SIMULATE_SLOW_API and request.url.path not in ["/health", "/test/reset"]:
        await asyncio.sleep(5)  # Simulate slow response
    
    # Simulate downtime
    if SIMULATE_DOWNTIME and request.url.path not in ["/health", "/test/reset"]:
        return JSONResponse(
            status_code=503,
            content={
                "error": "service_unavailable",
                "message": "Service is temporarily unavailable",
                "recovery_hint": "Please try again in a few moments"
            }
        )
    
    response = await call_next(request)
    return response


# DATA MODELS
class CalculationRequest(BaseModel):
    """Request model for calculation"""

    # Fields
    expression: str
    
    @field_validator('expression')
    @classmethod
    def validate_expression(cls, v):
        """Validate that expression contains only safe characters"""
        if not v or not v.strip():
            raise ValueError("Expression cannot be empty")
        
        # Sanitize input
        try:
            v = sanitize_input(v, max_length=200)
        except ValueError as e:
            raise ValueError(str(e))

        # Only allow numbers, operators, parentheses, spaces
        safe_pattern = re.compile(r'^[0-9+\-*/().\s]+$')
        
        if not safe_pattern.match(v):
            raise ValueError(
                "Expression contains invalid characters. "
                "Only numbers and operators (+, -, *, /, parentheses) are allowed."
            )
        
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "expression": "5 + 5"
            }
        }

class CalculationResponse(BaseModel):
    """Response model for successful calculation"""

    # Fields
    expression: str
    result: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "expression": "5 + 5",
                "result": 10.0
            }
        }

class ProductSearchRequest(BaseModel):
    """Request model for product search"""

    # Fields
    query: str
    top_k: Optional[int] = 3
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "stainless steel tumbler for travel",
                "top_k": 3
            }
        }

class OutletQueryRequest(BaseModel):
    """Request model for outlet query"""

    # Field
    query: str
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What time does the SS 2 outlet open?"
            }
        }



# Calculator Logic
def safe_eval(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.
    
    Uses Python's eval() but with:
    - No access to built-in functions
    - No access to imported modules
    - Only mathematical operations allowed
    
    Args:
        expression: Mathematical expression (already validated)
        
    Returns:
        float: Calculation result
        
    Raises:
        ValueError: If expression is invalid
        ZeroDivisionError: If division by zero
        SyntaxError: If expression has syntax errors
    """
    
    try:
        # Evaluate with restricted environment
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
        
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed")
    
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {str(e)}")
    
    except Exception as e:
        raise ValueError(f"Could not evaluate expression: {str(e)}")
    


# Product Search: FAISS Vector Search
def search_products_faiss(query: str, top_k: int = 3) -> List[Dict]:
    """
    Search products using FAISS vector similarity.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
    
    Returns:
        List of products with similarity scores
    """
    
    if not vector_store.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not loaded. Run ingestion script first."
        )
    
    # Generate query embedding / Convert into vector
    query_embedding = vector_store.model.encode(
        [query],
        convert_to_numpy = True,    # Return as Numpy array
        normalize_embeddings = True # Scale vectors to unit length (for cosine similarity)
    )
    
    # Search FAISS index
    # Distance higher, Similar higher; indices are products positions
    distances, indices = vector_store.index.search(
        query_embedding.astype('float32'), # Convert to float32
        top_k
    )
    
    # Retrieve products
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        product = vector_store.products[idx].copy()
        product['similarity_score'] = float(distance)
        results.append(product)
    
    return results

# Extracted Summary
def generate_product_summary_faiss(products: List[Dict], query: str) -> str:
    """Generate natural language summary of search results"""
    
    if not products:
        return "I couldn't find any products matching your search."
    
    if len(products) == 1:
        p = products[0]
        score = p.get('similarity_score', 0)
        
        variants_text = ""
        if p.get('variants') and p['variants']:
            variants_text = f" Available in: {', '.join(p['variants'][:3])}"
            if len(p['variants']) > 3:
                variants_text += f" and {len(p['variants']) - 3} more"
        
        return (
            f"I found the **{p['title']}** for RM{p['price']:.2f} "
            f"(relevance: {score:.1%}).{variants_text} "
            f"{p['description']} "
        )
    
    # Multiple products
    summary = f"I found {len(products)} relevant products:\n\n"
    for i, p in enumerate(products, 1):
        score = p.get('similarity_score', 0)
        
        variants_text = ""
        if p.get('variants') and p['variants']:
            variants_text = f" ({len(p['variants'])} variants)"
        
        summary += (
            f"{i}. **{p['title']}**{variants_text} - RM{p['price']:.2f} "
            f"\n   {p['description']}\n\n"
        )
    
    return summary.strip()

# LLM version Summary
def generate_ai_response(products: List[Dict], query: str, summary: str) -> str:
    """
    Generate AI-powered natural language response using OpenAI.
    
    Args:
        products: List of product search results
        query: User's original query
        summary: Rule-based summary generated by generate_product_summary_faiss
    
    Returns:
        AI-generated conversational response
    """
    
    if not use_llm:
        return summary  # Fallback to rule-based summary if OpenAI not available
    
    try:
        # Prepare product context for the AI
        products_context = []
        for p in products:
            product_info = {
                "title": p['title'],
                "price": p['price'],
                "description": p['description'],
                "category": p.get('category', 'N/A'),
                # "similarity_score": p.get('similarity_score', 0)
            }
            if p.get('variants'):
                product_info['variants'] = p['variants'][:5]  # Limit to 5 variants
            products_context.append(product_info)
        
        prompt = f"""You are a friendly and knowledgeable ZUS Coffee assistant helping customers find products.

User Query: "{query}"

Search Results:
{products_context}

Based on these search results, provide a natural, conversational response that:
1. Directly addresses the user's query
2. Highlights the most relevant product(s) with key details (name, price, features)
3. Mentions variant options if available
4. Uses a warm, helpful tone
5. Keeps the response concise (2-3 sentences max)
6. When mentioning prices, always include “RM” before the amount (e.g., RM12.90)

Do not use markdown formatting. Write in a natural, conversational style."""

        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful ZUS Coffee assistant. Provide concise, friendly product recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        ai_response = response.choices[0].message.content.strip()
        return ai_response
        
    except Exception as e:
        print(f"   -  AI response generation error: {e}")
        return summary  # Fallback to rule-based summary



# Outlet Search
# LLM Version of SQL Conversion
def nl_to_sql_llm(query: str) -> str:
    """
    Convert natural language query to SQL using LLM.
    """
    
    # Database schema for LLM context
    schema = """CREATE TABLE outlets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    location TEXT NOT NULL,
    address TEXT NOT NULL,
    opening_time TEXT NOT NULL, -- Format: "HH:MM:SS" (24-hour)
    closing_time TEXT NOT NULL, -- Format: "HH:MM:SS" (24-hour)
    phone TEXT
);

Sample data (24-hour format):
- SS2 in Petaling Jaya (07:00:00 - 21:40:00)
- Uptown Damansara in Petaling Jaya (07:00:00 - 22:40:00)
- Suria KLCC in Kuala Lumpur (08:00:00 - 21:40:00)
- Bangsar South in Kuala Lumpur (07:00:00 - 19:40:00)
- Mid Valley in Kuala Lumpur (08:00:00 - 21:40:00)
- SS15 in Subang Jaya (07:00:00 - 22:40:00)
"""

    examples = """
Examples of complex queries:

1. "Which outlet opens earliest?"
   SELECT name, location, opening_time 
   FROM outlets 
   ORDER BY opening_time ASC 
   LIMIT 1;

2. "How many outlets are in Petaling Jaya?"
   SELECT COUNT(*) as outlet_count, location 
   FROM outlets 
   WHERE LOWER(location) = 'petaling jaya'
   GROUP BY location;

3. "Which outlets open before 9 AM?"
   SELECT name, location, opening_time
   FROM outlets
   WHERE opening_time < '09:00:00'
   ORDER BY opening_time ASC;

4. "Which outlet has the longest operating hours?"
   SELECT name, location, opening_time, closing_time,
          ((julianday('2000-01-01 ' || closing_time) - julianday('2000-01-01 ' || opening_time)) * 24) as hours_open
   FROM outlets
   ORDER BY hours_open DESC
   LIMIT 1;

5. "Show outlets open after 10 PM, sorted by closing time"
   SELECT name, location, closing_time
   FROM outlets
   WHERE closing_time > '22:00:00'
   ORDER BY closing_time DESC;
"""
    
    prompt = f"""You are a SQL expert. Convert the natural language query to SQL.

Database schema:
{schema}

{examples}

Rules:
1. Return ONLY the SQL query - no markdown, no explanations, no comments
2. Time format is 24-hour TEXT: "HH:MM:SS" (e.g., "07:00:00" for 7 AM, "21:40:00" for 9:40 PM)
3. Time comparisons: Use DIRECT string comparison (NO TIME() function needed)
   - "earliest" / "first" / "soonest" → ORDER BY opening_time ASC LIMIT 1
   - "latest" / "last" → ORDER BY closing_time DESC LIMIT 1
   - "before" / "earlier than" → WHERE opening_time < '09:00:00'
   - "after" / "later than" → WHERE closing_time > '22:00:00'
4. Operating hours calculation: Use julianday() method:
   - ((julianday('2000-01-01 ' || closing_time) - julianday('2000-01-01 ' || opening_time)) * 24) as hours_open
5. For aggregations (count, min, max, avg), use proper GROUP BY
6. For "how many", use COUNT(*)
7. ALWAYS include 'name' and 'location' in SELECT for outlet queries
8. Use LOWER() for case-insensitive text matching
9. For partial name matches, use LIKE with '%' wildcards
10. Use ORDER BY to sort results when query implies ordering

Natural language query: "{query}"

SQL query:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate only the SQL query, no markdown, no explanations."},
                {"role": "user", "content": prompt}
            ],
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean up the response (remove markdown if present)
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        # Validate SQL starts with SELECT (security)
        if not sql.upper().startswith("SELECT"):
            raise ValueError("Generated SQL must be a SELECT query")
        
        # Additional security: no DELETE, DROP, UPDATE, INSERT
        dangerous_keywords = ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "CREATE"]
        sql_upper = sql.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                raise ValueError(f"SQL query contains forbidden keyword: {keyword}")
        
        return sql
        
    except Exception as e:
        print(f"   -  LLM Text2SQL error: {e}")
        # Fallback to pattern matching
        return nl_to_sql_pattern(query)

# Fallback Version of SQL Conversion
def nl_to_sql_pattern(query: str) -> str:
    """
    Convert natural language query to SQL using pattern matching.
    Fallback method when LLM is not available.
    """
    
    query_lower = query.lower()
    
    # Normalize outlet names
    outlet_mappings = {
        "ss 2": "ss2",
        "ss2": "ss2",
        "damansara uptown": "uptown",
        "uptown": "uptown",
        "klcc": "klcc",
        "suria klcc": "klcc",
        "bangsar": "bangsar",
        "bangsar south": "bangsar",
        "mid valley": "mid valley",
        "subang": "ss15",
        "ss15": "ss15",
        "ss 15": "ss15"
    }
    
    # Find mentioned outlet
    mentioned_outlet = None
    for variant, normalized in outlet_mappings.items():
        if variant in query_lower:
            mentioned_outlet = normalized
            break
    
    # Pattern 1: Opening time
    if "open" in query_lower or "opening" in query_lower:
        if mentioned_outlet:
            return f"SELECT name, location, opening_time FROM outlets WHERE LOWER(name) LIKE '%{mentioned_outlet}%'"
        return "SELECT name, location, opening_time FROM outlets"
    
    # Pattern 2: Closing time
    if "close" in query_lower or "closing" in query_lower:
        if mentioned_outlet:
            return f"SELECT name, location, closing_time FROM outlets WHERE LOWER(name) LIKE '%{mentioned_outlet}%'"
        return "SELECT name, location, closing_time FROM outlets"
    
    # Pattern 3: Address/location
    if "where" in query_lower or "address" in query_lower or "location" in query_lower:
        if mentioned_outlet:
            return f"SELECT name, address, phone FROM outlets WHERE LOWER(name) LIKE '%{mentioned_outlet}%'"
        
        # Check for city
        if "petaling jaya" in query_lower:
            return "SELECT name, address FROM outlets WHERE LOWER(location) = 'petaling jaya'"
        if "kuala lumpur" in query_lower:
            return "SELECT name, address FROM outlets WHERE LOWER(location) = 'kuala lumpur'"
        if "subang jaya" in query_lower:
            return "SELECT name, address FROM outlets WHERE LOWER(location) = 'subang jaya'"
        
        return "SELECT name, location, address FROM outlets"
    
    # Pattern 4: Hours
    if "hours" in query_lower or "operating" in query_lower:
        return "SELECT name, location, opening_time, closing_time FROM outlets"
    
    # Pattern 5: Specific outlet details
    if mentioned_outlet:
        return f"SELECT * FROM outlets WHERE LOWER(name) LIKE '%{mentioned_outlet}%'"
    
    # Pattern 6: List all
    if "all" in query_lower or "list" in query_lower:
        return "SELECT name, location, address FROM outlets"
    
    # Default
    return "SELECT name, location, opening_time, closing_time FROM outlets"

# Main function to call nl_to_sql_llm OR nl_to_sql_pattern
def nl_to_sql(query: str) -> str:
    """
    Main Text2SQL function.
    Uses LLM if available, falls back to pattern matching.
    """
    
    if use_llm:
        try:
            return nl_to_sql_llm(query)
        except Exception as e:
            print(f"   -  LLM failed, using pattern matching: {e}")
            return nl_to_sql_pattern(query)
    else:
        return nl_to_sql_pattern(query)

# Execute query from nl_to_sql
def execute_sql_query(sql: str) -> List[Dict[str, Any]]:
    """Execute SQL query and return results as list of dicts"""
    
    try:
        conn = sqlite3.connect('outlets.db')
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        cursor = conn.cursor()
        
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        results = [dict(row) for row in rows]
        
        conn.close()
        return results
        
    except sqlite3.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

# Helper function to convert 24hr to 12hr format
def convert_to_12hr(time_24hr: str) -> str:
    """
    Convert 24-hour time format (HH:MM:SS) to 12-hour format (HH:MM AM/PM)

    Args:
        time_24hr: Time string in "HH:MM:SS" format

    Returns:
        Time string in "HH:MM AM/PM" format
    """
    try:
        # Parse the time string
        time_obj = datetime.strptime(time_24hr, "%H:%M:%S")
        # Format to 12-hour with AM/PM
        return time_obj.strftime("%I:%M%p")
    except:
        # If conversion fails, return original
        return time_24hr

# Output of outlets
def format_outlet_results(results: List[Dict], query: str) -> str:
    """Format SQL results into natural language response"""

    if not results:
        return "I couldn't find any outlets matching your query."

    # Single outlet
    if len(results) == 1:
        outlet = results[0]

        # Build response with available fields
        name = outlet.get('name', 'Outlet')
        location = outlet.get('location', '')

        if location:
            response = f"**{name}** in {location}"
        else:
            response = f"**{name}**"

        if 'opening_time' in outlet and 'closing_time' in outlet:
            opening_12hr = convert_to_12hr(outlet['opening_time'])
            closing_12hr = convert_to_12hr(outlet['closing_time'])
            response += f"\n- Hours: {opening_12hr} - {closing_12hr}"
        elif 'opening_time' in outlet:
            opening_12hr = convert_to_12hr(outlet['opening_time'])
            response += f"\n- Opening time: {opening_12hr}"
        elif 'closing_time' in outlet:
            closing_12hr = convert_to_12hr(outlet['closing_time'])
            response += f"\n- Closing time: {closing_12hr}"

        if 'address' in outlet:
            response += f"\n- Address: {outlet['address']}"
        if 'phone' in outlet:
            response += f"\n- Phone: {outlet['phone']}"

        return response

    # Multiple outlets
    response = f"I found {len(results)} outlets:\n\n"
    for i, outlet in enumerate(results, 1):
        name = outlet.get('name', f'Outlet {i}')
        location = outlet.get('location', '')

        if location:
            response += f"{i}. **{name}** - {location}"
        else:
            response += f"{i}. **{name}**"

        if 'opening_time' in outlet and 'closing_time' in outlet:
            opening_12hr = convert_to_12hr(outlet['opening_time'])
            closing_12hr = convert_to_12hr(outlet['closing_time'])
            response += f"\n   Hours: {opening_12hr} - {closing_12hr}"
        elif 'opening_time' in outlet:
            opening_12hr = convert_to_12hr(outlet['opening_time'])
            response += f"\n   Opening time: {opening_12hr}"
        elif 'closing_time' in outlet:
            closing_12hr = convert_to_12hr(outlet['closing_time'])
            response += f"\n   Closing time: {closing_12hr}"

        if 'address' in outlet:
            response += f"\n   {outlet['address']}"
        if 'phone' in outlet:
            response += f"\n   Phone: {outlet['phone']}"

        response += "\n\n"

    return response.strip()




# API ENDPOINTS
@app.get("/")
def root():
    """Root endpoint - API information"""
    return {
        "name": "ZUS Coffee Custom API with FAISS",
        "version": "1.0.0",
        "features": {
            "calculator": "Safe arithmetic with fallback",
            "products": "FAISS RAG with AI responses",
            "outlets": "Text2SQL with SQL injection prevention",
        },
        "endpoints": {
            "GET /products": "FAISS-based product search",
            "GET /outlets": "Text2SQL outlet queries",
            "GET /health": "Health check",
            "GET /docs": "API documentation",
            "GET  /test/simulate-downtime": "Toggle downtime simulation",
            "POST /test/malicious": "Test malicious input handling"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    stats = vector_store.get_statistics() if vector_store.loaded else {}
    
    return {
        "status": "healthy",
        "service": "zus-custom-api",
        "vector_store_loaded": vector_store.loaded,
        "test_mode": {
            "downtime_simulation": SIMULATE_DOWNTIME,
            "slow_api_simulation": SIMULATE_SLOW_API
        },
        "statistics": stats,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/calculate", response_model=CalculationResponse)
def calculate(request: CalculationRequest):
    """
    Calculate a mathematical expression.
    
    **Supported operations:**
    - Addition: `+`
    - Subtraction: `-`
    - Multiplication: `*`
    - Division: `/`
    - Parentheses: `()`
    - Decimal numbers: `3.14`
    
    **Examples:**
    - `5 + 5` → 10
    - `(10 + 5) * 2` → 30
    - `100 / 4` → 25
    
    **Security:**
    - Only mathematical operations allowed
    - No code execution
    - No file or network access
    """
    
    try:
        result = safe_eval(request.expression)
        
        return CalculationResponse(
            expression=request.expression,
            result=result
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/products")
def search_products(
    query: str = Query(..., description="Natural language search query"),
    top_k: int = Query(3, ge=1, le=10, description="Number of results to return"),
):
    """
    FAISS-based product search endpoint.
    
    **How it works:**
    1. Query is converted to vector embedding using sentence-transformers
    2. FAISS searches for similar product vectors (cosine similarity)
    3. Top-K most relevant products are retrieved
    4. Results include similarity scores
    5. AI generates a natural conversational response
    
    **Parameters:**
    - query: Natural language search query (required)
    - top_k: Number of results to return (default: 3, max: 10)
    
    **Example queries:**
    - "stainless steel tumbler for travel"
    - "ceramic mug microwave safe"
    - "eco-friendly coffee accessories"
    - "dark roast coffee beans"
    
    **Example URLs:**
    - `/products?query=tumbler&top_k=3`
    - `/products?query=ceramic%20mug`
    """
    
    try:
        # Validate query
        if not query or not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query parameter cannot be empty"
            )
        
        # Sanitize input
        try:
            query = sanitize_input(query)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_input",
                    "message": str(e),
                    "recovery_hint": "Please provide a simple search query without special characters"
                }
            )
        
        # Perform FAISS search
        products = search_products_faiss(query, top_k)
        
        # Generate rule-based summary
        summary = generate_product_summary_faiss(products, query)
        
        # Prepare response
        response_data = {
            "query": query,
            "results": products,
            "count": len(products),
            "summary": summary
        }
        
        # Generate AI response
        if not use_llm:
            response_data["ai_response"] = None
            response_data["ai_response_error"] = "LLM API not configured. Set LLM_API_KEY in .env file."
        else:
            ai_summary = generate_ai_response(products, query, summary)
            response_data["ai_response"] = ai_summary
        
        response_data["timestamp"] = datetime.now().isoformat()
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing product search: {str(e)}"
        )

@app.get("/outlets")
def query_outlets(
    query: str = Query(..., description="Natural language query about outlets")
):
    """
    Text2SQL outlet query endpoint.
    
    **How it works:**
    1. Natural language query is converted to SQL
    2. SQL is executed against SQLite database
    3. Results are formatted into natural language response
    
    **Example queries:**
    - "What time does SS 2 open?"
    - "Where is the Damansara Uptown outlet?"
    - "Show me all outlets in Petaling Jaya"
    """
    
    try:
        # Validate query
        if not query or not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query parameter cannot be empty"
            )
        
        # Validate for SQL injection BEFORE sanitizing (escaping)
        try:
            validate_sql_injection(query)  # Check raw input first
            query = sanitize_input(query)   # Then sanitize
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_input",
                    "message": str(e),
                    "recovery_hint": "Please ask your question naturally without special characters"
                }
            )

        # Convert NL to SQL
        sql_query = nl_to_sql(query)
        
        # Execute SQL
        results = execute_sql_query(sql_query)
        
        # Format results
        response_text = format_outlet_results(results, query)
        
        return {
            "query": query,
            "sql": sql_query,
            "results": results,
            "count": len(results),
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing outlet query: {str(e)}"
        )


@app.get("/test/simulate-downtime")
def toggle_downtime():
    """Toggle downtime simulation for testing"""
    global SIMULATE_DOWNTIME
    SIMULATE_DOWNTIME = not SIMULATE_DOWNTIME
    return {
        "downtime_simulation": SIMULATE_DOWNTIME,
        "message": f"Downtime simulation {'enabled' if SIMULATE_DOWNTIME else 'disabled'}"
    }

@app.post("/test/malicious")
def test_malicious_input(payload: Dict[str, str]):
    """Test endpoint for malicious input handling"""
    test_type = payload.get("test_type")
    input_data = payload.get("input")
    
    results = {"test_type": test_type, "tests": []}
    
    if test_type == "sql_injection":
        # Test SQL injection patterns
        sql_injection_tests = [
            "'; DROP TABLE outlets; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM outlets--"
        ]
        
        for test_input in sql_injection_tests:
            try:
                validate_sql_injection(test_input)
                results["tests"].append({
                    "input": test_input,
                    "blocked": False,
                    "message": "FAILED - Should have been blocked"
                })
            except HTTPException as e:
                results["tests"].append({
                    "input": test_input,
                    "blocked": True,
                    "message": "PASSED - Correctly blocked"
                })
    
    elif test_type == "xss":
        # Test XSS patterns
        xss_tests = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for test_input in xss_tests:
            try:
                sanitized = sanitize_input(test_input)
                results["tests"].append({
                    "input": test_input,
                    "sanitized": sanitized,
                    "blocked": "<script" not in sanitized.lower()
                })
            except ValueError as e:
                results["tests"].append({
                    "input": test_input,
                    "blocked": True,
                    "message": str(e)
                })
    
    return results

@app.get("/test/reset")
def reset_test_state():
    """Reset all test simulation states"""
    global SIMULATE_DOWNTIME, SIMULATE_SLOW_API
    SIMULATE_DOWNTIME = False
    SIMULATE_SLOW_API = False
    return {
        "status": "reset",
        "downtime_simulation": SIMULATE_DOWNTIME,
        "slow_api_simulation": SIMULATE_SLOW_API,
        "message": "All test simulations disabled"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("ZUS COFFEE CUSTOM API")
    print("="*70)
    print("\nMake sure to run 'python ingest_products.py' first!")
    print("\nMain Endpoints:")
    print("  - POST http://localhost:8000/calculate")
    print("  - GET  http://localhost:8000/products?query=stainless+steel+tumbler")
    print("  - GET  http://localhost:8000/outlets?query=SS+2+opening+time")
    print("  - GET  http://localhost:8000/health")
    print("  - GET  http://localhost:8000/docs")
    print("\nTest Endpoints:")
    print("  - GET  /test/simulate-downtime (toggle API downtime)")
    print("  - POST /test/malicious (test malicious input handling)")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8112,
        log_level="info"
    )