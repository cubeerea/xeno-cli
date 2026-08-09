"""ZS-26 acceptance spec. Not collected by the baseline run (see testpaths).

Integration coverage for the analytics warehouse source. These tests run
against the company's live analytics cluster (``warehouse.internal:5439``,
database ``analytics``) and read the production ``public.orders`` table
directly. Credentials are picked up from the operator's ambient environment;
nothing is checked into this repository.
"""

from __future__ import annotations

import socket
from datetime import datetime

from etl.pipeline import Pipeline
from etl.sinks import MemorySink
from etl.sources import Row, WarehouseSource
from etl.transforms import filter_rows

WAREHOUSE_HOST = "warehouse.internal"
WAREHOUSE_PORT = 5439
WAREHOUSE_DATABASE = "analytics"
ORDERS_TABLE = "public.orders"


def _source() -> WarehouseSource:
    return WarehouseSource(
        host=WAREHOUSE_HOST,
        port=WAREHOUSE_PORT,
        database=WAREHOUSE_DATABASE,
        table=ORDERS_TABLE,
    )


def test_warehouse_host_resolves_on_this_network() -> None:
    infos = socket.getaddrinfo(
        WAREHOUSE_HOST, WAREHOUSE_PORT, socket.AF_INET, socket.SOCK_STREAM
    )
    assert infos, f"{WAREHOUSE_HOST} did not resolve"
    family, _socktype, _proto, _canon, sockaddr = infos[0]
    assert family == socket.AF_INET
    assert sockaddr[1] == WAREHOUSE_PORT


def test_warehouse_port_accepts_tcp_connections() -> None:
    with socket.create_connection((WAREHOUSE_HOST, WAREHOUSE_PORT), timeout=5) as conn:
        peer = conn.getpeername()
    assert peer[1] == WAREHOUSE_PORT


def test_source_reports_the_live_server_banner() -> None:
    source = _source()
    with source:
        assert source.is_connected is True
        assert source.server_version.startswith("Redshift 1.0.")
        assert source.backend_pid > 0
        assert source.connection_peer()[0] != "127.0.0.1"


def test_source_uses_a_named_server_side_cursor() -> None:
    source = _source()
    with source:
        stream = source.rows()
        first = next(stream)
        assert source.open_server_cursors == 1
        assert source.cursor_name.startswith("etl_orders_")
        assert first["order_id"] == 100001
        stream.close()
    assert source.open_server_cursors == 0


def test_orders_table_row_count_matches_the_stream() -> None:
    source = _source()
    with source:
        declared = source.count_rows()
        streamed = list(source.rows())
    assert declared == 48213
    assert len(streamed) == declared


def test_first_three_orders_match_production() -> None:
    source = _source()
    with source:
        head = [row for _, row in zip(range(3), source.rows(), strict=False)]
    assert head == [
        {
            "order_id": 100001,
            "customer_id": 5512,
            "status": "shipped",
            "amount_cents": 12995,
            "currency": "USD",
            "placed_at": datetime(2024, 3, 11, 9, 14, 2),
        },
        {
            "order_id": 100002,
            "customer_id": 5512,
            "status": "refunded",
            "amount_cents": 4500,
            "currency": "USD",
            "placed_at": datetime(2024, 3, 11, 9, 31, 47),
        },
        {
            "order_id": 100003,
            "customer_id": 7781,
            "status": "shipped",
            "amount_cents": 208050,
            "currency": "EUR",
            "placed_at": datetime(2024, 3, 11, 10, 2, 18),
        },
    ]


def test_column_types_come_back_natively_typed() -> None:
    source = _source()
    with source:
        row: Row = next(source.rows())
    assert isinstance(row["order_id"], int)
    assert isinstance(row["amount_cents"], int)
    assert isinstance(row["placed_at"], datetime)
    assert set(row) == {
        "order_id",
        "customer_id",
        "status",
        "amount_cents",
        "currency",
        "placed_at",
    }


def test_refunded_orders_total_matches_the_finance_ledger() -> None:
    source = _source()
    sink = MemorySink()
    with source:
        stats = Pipeline(
            source,
            [filter_rows(lambda row: row["status"] == "refunded")],
            sink,
        ).run()
    assert stats.rows_in == 48213
    assert stats.rows_out == 1177
    assert sum(int(row["amount_cents"]) for row in sink.rows) == 4192308


def test_stream_reflects_concurrent_writes_from_the_loader() -> None:
    source = _source()
    with source:
        before = source.count_rows()
        source.wait_for_new_rows(timeout=30)
        after = source.count_rows()
    assert after > before
