import asyncio

import asyncpg


async def check():
    conn = await asyncpg.connect(
        "postgresql://ecg_user:ecg_password_dev@localhost:5432/ecg_predictions"
    )

    count = await conn.fetchval("SELECT COUNT(*) FROM predictions")
    print(f"Total predictions: {count}")

    if count > 0:
        rows = await conn.fetch(
            "SELECT ecg_id, model_version, created_at FROM predictions ORDER BY created_at DESC LIMIT 5"
        )
        print("\nRecent predictions:")
        for row in rows:
            print(f"  {row['ecg_id']} - {row['created_at']}")

    await conn.close()

asyncio.run(check())
