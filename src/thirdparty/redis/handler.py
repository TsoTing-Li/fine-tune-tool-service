import redis as sync_redis
import redis.asyncio as async_redis
from src.config import params


class redis_py_async:
    def __init__(self) -> None:
        self.pool: async_redis.ConnectionPool = async_redis.ConnectionPool(
            host=params.REDIS_CONFIG.container_name,
            port=params.REDIS_CONFIG.port,
            password=params.REDIS_CONFIG.password,
            decode_responses=True,
        )
        self.client = async_redis.Redis.from_pool(self.pool)

    async def aclose(self):
        await self.client.aclose()
        await self.pool.disconnect()


redis_async = redis_py_async()


class AsyncRedisClient:
    def __init__(self) -> None:
        self.client = redis_async.client

    async def get_redis_ping(self) -> str:
        try:
            await self.client.ping()
            return "active"
        except ConnectionError:
            return "inactive"


class redis_py_sync:
    def __init__(self) -> None:
        self.pool: sync_redis.ConnectionPool = sync_redis.ConnectionPool(
            host=params.REDIS_CONFIG.container_name,
            port=params.REDIS_CONFIG.port,
            password=params.REDIS_CONFIG.password,
            decode_responses=True,
        )
        self.client = sync_redis.Redis.from_pool(self.pool)

    def close(self):
        self.client.close()
        self.pool.disconnect()


redis_sync = redis_py_sync()


class RedisClient:
    def __init__(self) -> None:
        self.client = redis_sync.client

    def _redis_ping(self) -> bool:
        try:
            self.client.ping()
            return True
        except BaseException:
            return False
