import pandas as pd
import sqlite3
import os

def csv_to_sqlite(csv_file_path, sqlite_db_path, table_name="data"):
    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return

    # Step 1: Read CSV into a DataFrame
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read CSV file with {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Step 2: Ensure the 'Date' column is in the correct format and add Formatted_Date
    if 'Date' in df.columns:
        try:
            # Convert 'Date' column to datetime (invalid dates become NaT)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            print("Successfully converted 'Date' column to datetime.")

            # Create the 'Formatted_Date' column with the format 'Month-Year' (e.g., December-2023)
            df['Formatted_Date'] = df['Date'].dt.strftime('%B-%Y')
            print("Added 'Formatted_Date' column with format 'Month-Year'.")
        except Exception as e:
            print(f"Error formatting 'Date' column: {e}")
    else:
        print("Warning: 'Date' column not found in the CSV file.")

    # Step 3: Clean 'Debit' column to remove negative signs
    if 'Debit' in df.columns:
        try:
            # Remove '-' and convert to float
            df['Debit'] = df['Debit'].astype(str).str.replace('-', '').astype(float)
            print("Successfully cleaned the 'Debit' column by removing negative signs.")
        except Exception as e:
            print(f"Error cleaning 'Debit' column: {e}")
    else:
        print("Warning: 'Debit' column not found in the CSV file.")

    # Step 4: Create or connect to SQLite database
    try:
        conn = sqlite3.connect(sqlite_db_path)
        print(f"Connected to SQLite database at '{sqlite_db_path}'.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    # Step 5: Write DataFrame to SQLite database
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Successfully wrote cleaned data to table '{table_name}' in the SQLite database.")
    except Exception as e:
        print(f"Error writing to SQLite database: {e}")
    finally:
        # Close the connection
        conn.close()
        print("Connection to database closed.")

if __name__ == "__main__":
    # User input: path to the CSV file and SQLite database file
    csv_path = input("Enter the path to the CSV file: ").strip()
    sqlite_path = input("Enter the desired SQLite database file path: ").strip()
    table_name = input("Enter the table name (default: 'data'): ").strip() or "data"

    # Convert CSV to SQLite
    csv_to_sqlite(csv_path, sqlite_path, table_name)
