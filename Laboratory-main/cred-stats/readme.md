
# CreditStatsBot

**CreditStatsBot** is a Streamlit-based chatbot that integrates SQL database querying capabilities with a memory system. 
It allows users to query their expense data stored in a SQL database and retains conversation context to provide a seamless 
experience. The memory can either be stored **in-memory** for local development or **persisted in Redis** for scalability.

## Features
- **SQL Query Integration**: Uses LangChain's `create_sql_agent` to perform SQL queries on a database.
- **Conversation Memory**: Supports both in-memory storage (default) and Redis for persistent chat histories.
- **User-Friendly UI**: Built with Streamlit for a clean and interactive experience.
- **Configurable Storage**: Easily switch between in-memory and Redis memory by setting environment variables.

## Requirements
- Python 3.8+
- Redis (optional for persistent memory)
- Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/creditstatsbot.git
   cd creditstatsbot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root directory and add the following:
     ```
     OPENAI_API_KEY=your_openai_api_key
     USE_REDIS=true  # Set to false for in-memory storage
     REDIS_URL=redis://localhost:6379/0
     ```

5. Start Redis (if using Redis):
   ```bash
   docker run -d -p 6379:6379 redis/redis-stack:latest
   ```

6. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- **Input Field**: Type questions about your expenses, e.g., "Where did I spend the most money in March 2024?"
- **Response**: The bot queries the database and provides the results.
- **Memory**: The conversation context is remembered across queries.

## Example

- **User**: "Where did I spend most money in April 2024?"
- **Bot**: "In April 2024, you spent the most money at KOMBI HAUS MOTORS with a total expenditure of $1,488.82."

## Configurable Memory Storage

- **In-Memory**: Default configuration for local development.
- **Redis**: Use Redis for persistent memory across sessions by setting `USE_REDIS=true` and providing the Redis URL.

## License

This project is licensed under the MIT License.