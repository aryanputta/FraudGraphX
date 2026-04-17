"""
Generate synthetic banking transaction dataset that includes:
  - Normal transactions
  - Fraud rings
  - Mule accounts
  - Synthetic identity fraud
  - Money laundering chains
"""

from __future__ import annotations

import json
import math
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
rng = np.random.default_rng(42)

OUTPUT_DIR = Path(__file__).parent.parent / "sample"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_CUSTOMERS = 5_000
NUM_MERCHANTS = 500
NUM_DEVICES = 3_000
NUM_TRANSACTIONS = 200_000
FRAUD_RATE = 0.015


# ── Seed entities ──────────────────────────────────────────────────────────────

def gen_customers(n: int) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        is_synthetic = rng.random() < 0.02          # 2% synthetic identities
        rows.append(
            {
                "customer_id": str(uuid.uuid4()),
                "name": fake.name(),
                "email": fake.email(),
                "phone": fake.phone_number(),
                "dob": fake.date_of_birth(minimum_age=18, maximum_age=85).isoformat(),
                "address": fake.address().replace("\n", ", "),
                "account_age_days": int(rng.integers(1, 3650)),
                "is_synthetic": is_synthetic,
                "avg_monthly_spend": float(rng.lognormal(6, 1.2)),
            }
        )
    return pd.DataFrame(rows)


def gen_merchants(n: int) -> pd.DataFrame:
    categories = [
        "grocery", "gas_station", "restaurant", "online_retail",
        "electronics", "travel", "gambling", "crypto_exchange",
        "jewelry", "pharmacy", "clothing",
    ]
    high_risk = {"gambling", "crypto_exchange", "jewelry"}
    rows = []
    for _ in range(n):
        cat = rng.choice(categories)
        rows.append(
            {
                "merchant_id": str(uuid.uuid4()),
                "name": fake.company(),
                "category": cat,
                "country": fake.country_code(),
                "risk_score": float(rng.uniform(0.5, 1.0) if cat in high_risk else rng.uniform(0.0, 0.4)),
            }
        )
    return pd.DataFrame(rows)


def gen_devices(n: int) -> pd.DataFrame:
    rows = [{"device_id": str(uuid.uuid4()), "device_type": rng.choice(["mobile", "desktop", "tablet"])} for _ in range(n)]
    return pd.DataFrame(rows)


# ── Transaction generators ─────────────────────────────────────────────────────

def normal_txn(customers: pd.DataFrame, merchants: pd.DataFrame, devices: pd.DataFrame, base_ts: datetime) -> Dict[str, Any]:
    cust = customers.sample(1).iloc[0]
    merch = merchants.sample(1).iloc[0]
    dev = devices.sample(1).iloc[0]
    ts = base_ts + timedelta(seconds=int(rng.integers(0, 86400)))
    return {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": ts.isoformat(),
        "amount": float(round(rng.lognormal(3.5, 1.0), 2)),
        "currency": "USD",
        "customer_id": cust["customer_id"],
        "account_id": f"acc_{cust['customer_id'][:8]}",
        "merchant_id": merch["merchant_id"],
        "merchant_category": merch["category"],
        "device_id": dev["device_id"],
        "ip_address": fake.ipv4(),
        "geolocation": {"lat": float(fake.latitude()), "lon": float(fake.longitude())},
        "card_type": rng.choice(["visa", "mastercard", "amex"]),
        "is_online": bool(rng.random() > 0.4),
        "channel": rng.choice(["web", "mobile", "pos"]),
        "is_fraud": False,
        "fraud_type": None,
    }


def fraud_ring_txns(customers: pd.DataFrame, merchants: pd.DataFrame, devices: pd.DataFrame, base_ts: datetime) -> List[Dict[str, Any]]:
    """A small cluster of customers share devices and target same merchant."""
    ring_customers = customers.sample(min(5, len(customers)))
    shared_device = devices.sample(1).iloc[0]["device_id"]
    target_merchant = merchants[merchants["category"].isin(["electronics", "jewelry"])].sample(1).iloc[0]
    txns = []
    for _, cust in ring_customers.iterrows():
        for _ in range(rng.integers(3, 8)):
            ts = base_ts + timedelta(minutes=int(rng.integers(0, 120)))
            txns.append(
                {
                    "transaction_id": str(uuid.uuid4()),
                    "timestamp": ts.isoformat(),
                    "amount": float(round(rng.uniform(800, 3000), 2)),
                    "currency": "USD",
                    "customer_id": cust["customer_id"],
                    "account_id": f"acc_{cust['customer_id'][:8]}",
                    "merchant_id": target_merchant["merchant_id"],
                    "merchant_category": target_merchant["category"],
                    "device_id": shared_device,
                    "ip_address": fake.ipv4_private(),
                    "geolocation": {"lat": float(fake.latitude()), "lon": float(fake.longitude())},
                    "card_type": "visa",
                    "is_online": True,
                    "channel": "web",
                    "is_fraud": True,
                    "fraud_type": "fraud_ring",
                }
            )
    return txns


def mule_account_txns(customers: pd.DataFrame, merchants: pd.DataFrame, devices: pd.DataFrame, base_ts: datetime) -> List[Dict[str, Any]]:
    """Large inbound transfers immediately followed by outbound to crypto."""
    mule = customers.sample(1).iloc[0]
    crypto_merch = merchants[merchants["category"] == "crypto_exchange"].sample(1).iloc[0] if (merchants["category"] == "crypto_exchange").any() else merchants.sample(1).iloc[0]
    txns = []
    for _ in range(rng.integers(2, 5)):
        ts_in = base_ts + timedelta(minutes=int(rng.integers(0, 30)))
        ts_out = ts_in + timedelta(minutes=int(rng.integers(1, 10)))
        amount = float(round(rng.uniform(5000, 50000), 2))
        txns += [
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": ts_in.isoformat(),
                "amount": amount,
                "currency": "USD",
                "customer_id": mule["customer_id"],
                "account_id": f"acc_{mule['customer_id'][:8]}",
                "merchant_id": merchants.sample(1).iloc[0]["merchant_id"],
                "merchant_category": "wire_transfer",
                "device_id": devices.sample(1).iloc[0]["device_id"],
                "ip_address": fake.ipv4(),
                "geolocation": {"lat": float(fake.latitude()), "lon": float(fake.longitude())},
                "card_type": "bank_transfer",
                "is_online": True,
                "channel": "web",
                "is_fraud": True,
                "fraud_type": "mule_account",
            },
            {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": ts_out.isoformat(),
                "amount": amount * float(rng.uniform(0.85, 0.98)),
                "currency": "USD",
                "customer_id": mule["customer_id"],
                "account_id": f"acc_{mule['customer_id'][:8]}",
                "merchant_id": crypto_merch["merchant_id"],
                "merchant_category": "crypto_exchange",
                "device_id": devices.sample(1).iloc[0]["device_id"],
                "ip_address": fake.ipv4(),
                "geolocation": {"lat": float(fake.latitude()), "lon": float(fake.longitude())},
                "card_type": "bank_transfer",
                "is_online": True,
                "channel": "web",
                "is_fraud": True,
                "fraud_type": "money_laundering",
            },
        ]
    return txns


# ── Main generation ────────────────────────────────────────────────────────────

def generate(n_transactions: int = NUM_TRANSACTIONS) -> None:
    print("Generating entities…")
    customers = gen_customers(NUM_CUSTOMERS)
    merchants = gen_merchants(NUM_MERCHANTS)
    devices = gen_devices(NUM_DEVICES)

    customers.to_csv(OUTPUT_DIR / "customers.csv", index=False)
    merchants.to_csv(OUTPUT_DIR / "merchants.csv", index=False)
    devices.to_csv(OUTPUT_DIR / "devices.csv", index=False)

    print("Generating transactions…")
    base_ts = datetime(2024, 1, 1)
    transactions: List[Dict[str, Any]] = []
    n_fraud_target = int(n_transactions * FRAUD_RATE)

    # Inject fraud patterns
    while len([t for t in transactions if t["is_fraud"]]) < n_fraud_target:
        day_offset = rng.integers(0, 365)
        ts = base_ts + timedelta(days=int(day_offset))
        fraud_choice = rng.choice(["ring", "mule"])
        if fraud_choice == "ring":
            transactions.extend(fraud_ring_txns(customers, merchants, devices, ts))
        else:
            transactions.extend(mule_account_txns(customers, merchants, devices, ts))

    # Fill normal transactions
    n_normal = n_transactions - len(transactions)
    for i in range(n_normal):
        day_offset = rng.integers(0, 365)
        ts = base_ts + timedelta(days=int(day_offset))
        transactions.append(normal_txn(customers, merchants, devices, ts))

    rng.shuffle(transactions)
    df = pd.DataFrame(transactions[:n_transactions])
    df.to_csv(OUTPUT_DIR / "transactions.csv", index=False)
    df.to_parquet(OUTPUT_DIR / "transactions.parquet", index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"Generated {len(df):,} transactions | fraud={fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    print(f"Output → {OUTPUT_DIR}")


if __name__ == "__main__":
    generate()
