from typing import Optional
import enum
import datetime as dt
from sqlalchemy import DateTime, ForeignKey, func, text
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy import String
from sqlalchemy.engine import Connection, Engine
from sqlalchemy import insert, select, update, delete
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import Enum
from tqdm import tqdm

metadata_obj = MetaData()
employee_table = Table(
    "employee",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("name", String(32), nullable=False, unique=True),
    Column("email_address", String(60), nullable=True),
)


class KrabbyPattyItemType(enum.Enum):
    KrabbyPatty = 1
    CoralBites = 2
    KelpRings = 3
    Special = 4


class Status(enum.Enum):
    Pending = 1
    InProgress = 2
    Done = 3
    Cancelled = 4


orders = Table(
    "orders",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("item_type", Enum(KrabbyPattyItemType), nullable=False),
    Column("status", Enum(Status), nullable=False, default=Status.Pending),
    Column("made_by", Integer, ForeignKey("employee.id"), nullable=False),
    Column("timestamp", DateTime, nullable=False, server_default=func.now()),
    Column("item_details", JSONB, nullable=True),
)


def create_tables(conn: Connection):
    metadata_obj.create_all(conn)


def add_employee(
    conn: Connection, name: str, email_address: Optional[str] = None
) -> int:
    stmt = insert(employee_table).values(name=name, email_address=email_address)
    result = conn.execute(stmt)
    conn.commit()

    return result.inserted_primary_key[0]


def add_order(
    conn: Connection,
    employee_id: int,
    item_type: KrabbyPattyItemType,
    item_details: Optional[dict] = None,
) -> int:
    stmt = insert(orders).values(
        made_by=employee_id, item_type=item_type, item_details=item_details
    )
    result = conn.execute(stmt)
    conn.commit()

    return result.inserted_primary_key[0]


def choose_dt(i: int):
    return dt.datetime(1999, 5, 1) + dt.timedelta(seconds=i * 100000)


def create_dataset(conn: Connection):
    spongebob_id = add_employee(conn, "SpongeBob", "spongebob@bikinibottom.io")
    squidward_id = add_employee(conn, "Squidward", "squidward@bikinibottom.io")
    mr_krabs_id = add_employee(conn, "Mr. Krabs", "mrkrabs@bikinibottom.io")

    for i in tqdm(range(22_000), desc="Inserting Krabby Patties"):
        rows = [
            {
                "made_by": spongebob_id,
                "status": Status.Done if i % 2 == 0 else Status.InProgress,
                "item_type": KrabbyPattyItemType.KrabbyPatty,
                "timestamp": choose_dt(i) + j * dt.timedelta(seconds=10),
            }
            for j in range(100)
        ]

        stmt = insert(orders)
        result = conn.execute(stmt, rows)
        conn.commit()

    for i in tqdm(range(22_000), desc="Inserting Coral Bites"):
        rows = [
            {
                "made_by": spongebob_id,
                "status": Status.Done if i % 2 == 0 else Status.InProgress,
                "item_type": KrabbyPattyItemType.CoralBites,
                "timestamp": choose_dt(i) + j * dt.timedelta(seconds=10),
            }
            for j in range(100)
        ]

        stmt = insert(orders)
        result = conn.execute(stmt, rows)
        conn.commit()

    stmt = insert(orders).values(
        made_by=spongebob_id,
        status=Status.InProgress,
        item_type=KrabbyPattyItemType.Special,
        item_details={"name": "Krusty Krab Pizza", "ingredients": []},
    )
    result = conn.execute(stmt)

    stmt = insert(orders).values(
        made_by=spongebob_id,
        status=Status.InProgress,
        item_type=KrabbyPattyItemType.Special,
        item_details={"name": "Triple Krabby Supreme", "ingredients": []},
    )
    result = conn.execute(stmt)

    conn.commit()

    stmt = (
        select(orders)
        .join(employee_table)
        .where(orders.c.item_type == KrabbyPattyItemType.Special)
    )
    result = conn.execute(stmt)
    for row in result:
        print(row)


def query(conn: Connection):
    stmt = (
        select(orders)
        .join(employee_table)
        .where(orders.c.item_type == KrabbyPattyItemType.Special)
    )
    result = conn.execute(stmt)
    return list(result)


def main():
    engine = create_engine(
        "postgresql+psycopg://postgres:pass1@localhost/postgres", echo=False
    )

    with engine.connect() as conn:
        create_tables(conn)
        create_dataset(conn)
        results = query(conn)
        for result in results:
            print(result)


if __name__ == "__main__":
    main()
