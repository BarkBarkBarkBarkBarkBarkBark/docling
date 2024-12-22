import os
import sqlite3
import pandas as pd
import json

# Database path
db_path = "unified.db"

# Table creation queries
create_invoices_table = """
CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_number TEXT NOT NULL,
    order_date DATE,
    shipped_date DATE,
    shipping_name TEXT,
    shipping_address_line1 TEXT,
    shipping_city TEXT,
    shipping_state TEXT,
    shipping_zip_code TEXT,
    shipping_country TEXT,
    billing_name TEXT,
    billing_address_line1 TEXT,
    billing_city TEXT,
    billing_state TEXT,
    billing_zip_code TEXT,
    billing_country TEXT,
    subtotal REAL,
    discounts REAL,
    shipping_handling REAL,
    tax REAL,
    grand_total REAL,
    payment_method TEXT,
    transaction_date DATE,
    payment_amount REAL,
    shipping_speed TEXT
);
"""

create_invoice_items_table = """
CREATE TABLE IF NOT EXISTS invoice_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id INTEGER,
    description TEXT,
    quantity INTEGER,
    price REAL,
    condition TEXT,
    FOREIGN KEY (invoice_id) REFERENCES invoices (id) ON DELETE CASCADE
);
"""

create_statements_table = """
CREATE TABLE IF NOT EXISTS statements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_date DATE,
    description TEXT,
    debit REAL,
    credit REAL,
    invoice_id INTEGER,
    match_found BOOLEAN DEFAULT 0,
    FOREIGN KEY (invoice_id) REFERENCES invoices (id)
);
"""

# Initialize database
def create_database():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(create_invoices_table)
        cursor.execute(create_invoice_items_table)
        cursor.execute(create_statements_table)
        print("Invoices, invoice_items, and statements tables created.")

# Function to populate invoices and invoice_items
json_input_dir = "json_outputs"
def populate_invoices():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for file in os.listdir(json_input_dir):
            if file.endswith(".json"):
                json_file_path = os.path.join(json_input_dir, file)
                print(f"Processing file: {json_file_path}")

                with open(json_file_path, "r") as f:
                    try:
                        data = json.load(f)

                        invoice_data = (
                            data.get("order_number"),
                            pd.to_datetime(data.get("order_date"), errors="coerce").strftime("%Y-%m-%d") if data.get("order_date") else None,
                            pd.to_datetime(data.get("shipped_date"), errors="coerce").strftime("%Y-%m-%d") if data.get("shipped_date") else None,
                            data["shipping_address"].get("name"),
                            data["shipping_address"].get("address_line_1"),
                            data["shipping_address"].get("city"),
                            data["shipping_address"].get("state"),
                            data["shipping_address"].get("zip_code"),
                            data["shipping_address"].get("country"),
                            data["billing_address"].get("name"),
                            data["billing_address"].get("address_line_1"),
                            data["billing_address"].get("city"),
                            data["billing_address"].get("state"),
                            data["billing_address"].get("zip_code"),
                            data["billing_address"].get("country"),
                            data["totals"].get("subtotal"),
                            data["totals"].get("discounts"),
                            data["totals"].get("shipping_handling"),
                            data["totals"].get("tax"),
                            data["totals"].get("grand_total"),
                            data["payment_information"].get("method"),
                            pd.to_datetime(data["payment_information"].get("transaction_date"), errors="coerce").strftime("%Y-%m-%d") if data["payment_information"].get("transaction_date") else None,
                            data["payment_information"].get("amount"),
                            data.get("shipping_speed")
                        )

                        cursor.execute("""
                            INSERT INTO invoices (
                                order_number, order_date, shipped_date, shipping_name,
                                shipping_address_line1, shipping_city, shipping_state, shipping_zip_code, shipping_country,
                                billing_name, billing_address_line1, billing_city, billing_state, billing_zip_code, billing_country,
                                subtotal, discounts, shipping_handling, tax, grand_total, payment_method, transaction_date, payment_amount, shipping_speed
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, invoice_data)

                        invoice_id = cursor.lastrowid

                        for item in data.get("items", []):
                            item_data = (
                                invoice_id,
                                item.get("description"),
                                item.get("quantity"),
                                item.get("price"),
                                item.get("condition")
                            )
                            cursor.execute("""
                                INSERT INTO invoice_items (
                                    invoice_id, description, quantity, price, condition
                                ) VALUES (?, ?, ?, ?, ?)
                            """, item_data)

                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON file {file}: {e}")
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")

        conn.commit()
        print("Invoices and items populated.")

# Function to clean and map credit card statements
def clean_and_map_statements(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Standardize headers to match the database schema
    column_mapping = {
        "Date": "transaction_date",
        "Description": "description",
        "Debit": "debit",
        "Credit": "credit"
    }
    df.rename(columns=column_mapping, inplace=True)

    # Standardize date format
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce").dt.date

    # Ensure numeric values for debit and credit
    if "debit" in df.columns:
        df["debit"] = pd.to_numeric(df["debit"], errors="coerce")
    if "credit" in df.columns:
        df["credit"] = pd.to_numeric(df["credit"], errors="coerce")

    # Add placeholder columns for matching
    df["invoice_id"] = None
    df["match_found"] = False

    return df

# Function to perform lookup and update matches
def match_statements_to_invoices(cleaned_df):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        for index, row in cleaned_df.iterrows():
            if pd.notnull(row["transaction_date"]) and pd.notnull(row["debit"]):
                cursor.execute("""
                    SELECT id FROM invoices
                    WHERE transaction_date = ? AND payment_amount = ?
                """, (row["transaction_date"], row["debit"]))

                result = cursor.fetchone()
                if result:
                    cleaned_df.at[index, "invoice_id"] = result[0]
                    cleaned_df.at[index, "match_found"] = True

        # Save updated data back to the statements table
        cleaned_df.to_sql("statements", conn, if_exists="replace", index=False)
        print("Statements table updated with matches.")

if __name__ == "__main__":
    create_database()
    populate_invoices()
    csv_path = input("Enter credit card statement CSV path: ").strip()
    cleaned_df = clean_and_map_statements(csv_path)
    match_statements_to_invoices(cleaned_df)
    print("Credit card statement processed and matched.")
