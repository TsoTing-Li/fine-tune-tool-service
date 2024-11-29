import asyncio
import os

import redis
import redis.asyncio as async_redis


class redis_async_pool:
    def __init__(self) -> None:
        self.pool: async_redis.ConnectionPool = async_redis.ConnectionPool(
            host=os.environ["REDIS_HOST"],
            port=os.environ["REDIS_PORT"],
            password=os.environ["REDIS_PASSWORD"],
        )
        self.client = async_redis.Redis.from_pool(self.pool)

    async def aclose(self):
        await self.client.aclose()
        await self.pool.disconnect()


class AsyncRedisClient:
    def __init__(self) -> None:
        self.client = redis_async_pool().client

    async def get_redis_ping(self) -> str:
        try:
            await self.client.ping()
            return "active"
        except ConnectionError:
            return "inactive"

    async def publish_msg(self, channel: str, msg: str):
        try:
            await self.client.publish(channel=channel, message=msg)
        except redis.ConnectionError as conn_err:
            raise ConnectionError(f"Redis connection error: {conn_err}") from None
        except redis.RedisError as redis_err:
            raise RuntimeError(f"Occurred unexpected error: {redis_err}") from None


async def test_status():
    async_redis = AsyncRedisClient()
    status = await async_redis.get_redis_ping()
    print(f"Redis status: {status}")
    await async_redis.client.aclose()


async def test_pub():
    async_redis = AsyncRedisClient()
    test_chn = "test_chn"
    test_msg = "test_msg"
    await async_redis.publish_msg(channel=test_chn, msg=test_msg)
    await async_redis.publish_msg(channel=test_chn, msg="exit")
    await async_redis.client.aclose()


if __name__ == "__main__":
    # asyncio.run(test_status())
    asyncio.run(test_pub())
