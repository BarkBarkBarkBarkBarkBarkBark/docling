import os
import sqlite3
import json

# Percorso del database SQLite
db_path = "MDFinvoices.db"

# Percorso della cartella con i file JSON
json_input_dir = "json_outputs"

# Query SQL per creare la tabella 'invoices'
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

# Query SQL per creare la tabella 'invoice_items'
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

# Funzione per creare il database e le tabelle
def create_database():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(create_invoices_table)
        cursor.execute(create_invoice_items_table)
        print("Database and tables created successfully.")

# Funzione per popolare i dati JSON nel database
def populate_database():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Itera su tutti i file JSON nella cartella
        for file in os.listdir(json_input_dir):
            if file.endswith(".json"):
                json_file_path = os.path.join(json_input_dir, file)
                print(f"Processing file: {json_file_path}")
                
                with open(json_file_path, "r") as f:
                    try:
                        data = json.load(f)
                        
                        # Inserisci i dati nella tabella 'invoices'
                        invoice_data = (
                            data.get("order_number"),
                            data.get("order_date"),
                            data.get("shipped_date"),
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
                            data["payment_information"].get("transaction_date"),
                            data["payment_information"].get("amount"),
                            data.get("shipping_speed")
                        )
                        
                        cursor.execute("""
                            INSERT INTO invoices (
                                order_number, order_date, shipped_date,
                                shipping_name, shipping_address_line1, shipping_city,
                                shipping_state, shipping_zip_code, shipping_country,
                                billing_name, billing_address_line1, billing_city,
                                billing_state, billing_zip_code, billing_country,
                                subtotal, discounts, shipping_handling, tax, grand_total,
                                payment_method, transaction_date, payment_amount, shipping_speed
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, invoice_data)
                        
                        invoice_id = cursor.lastrowid  # Ottieni l'ID dell'invoice appena inserito
                        
                        # Inserisci i dati nella tabella 'invoice_items'
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

        # Commit delle modifiche
        conn.commit()
        print("Data inserted successfully.")

# Esegui il codice
if __name__ == "__main__":
    # Crea il database e le tabelle
    create_database()

    # Popola il database con i dati JSON
    populate_database()
