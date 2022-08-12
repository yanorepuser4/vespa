package com.yahoo.jdisc.http.server.jetty;

import org.eclipse.jetty.io.ByteBufferPool;
import org.eclipse.jetty.util.BufferUtil;

import java.nio.ByteBuffer;

class ThreadLocalByteBufferPool implements ByteBufferPool {
    static int bufferId(int size) {
        return 32 - Integer.numberOfLeadingZeros(size);
    }
    private static class BufferList {
        final ByteBuffer [] buffers = new ByteBuffer[16];
        int latest = -1;
        ByteBuffer acquire() {
            if (latest == -1) return null;
            ByteBuffer buf = buffers[latest];
            buffers[latest] = null;
            latest--;
            return buf;
        }
        ByteBuffer release(ByteBuffer buf) {
            if (latest >= (buffers.length - 1)) return buf;
            buffers[++latest] = buf;
            BufferUtil.clearToFill(buf);
            return null;
        }
    }
    private static class Cache {
        private int cachedBytes = 0;
        private final BufferList [] direct;
        private final BufferList [] heap;
        Cache(int numBufferClasses) {
            direct = new BufferList[numBufferClasses];
            heap = new BufferList[numBufferClasses];
            for (int i=0; i < numBufferClasses; i++) {
                direct[i] = new BufferList();
                heap[i] = new BufferList();
            }
        }
        ByteBuffer acquire(int bufferId, boolean direct) {
            ByteBuffer buf =  direct ? this.direct[bufferId].acquire() : this.heap[bufferId].acquire();
            if (buf != null)
                cachedBytes -= buf.capacity();
            return buf;
        }
        ByteBuffer release(int bufferId, ByteBuffer buf) {
            ByteBuffer overflow = buf.isDirect() ? direct[bufferId].release(buf) : heap[bufferId].release(buf);
            if (overflow == null)
                cachedBytes += buf.capacity();
            return overflow;
        }
    }
    private static class ThreadLocalCache extends ThreadLocal<Cache> {
        private final int numBufferClasses;
        ThreadLocalCache(int numBufferClasses) {
            this.numBufferClasses = numBufferClasses;
        }
        @Override
        protected Cache initialValue() {
            return new Cache(numBufferClasses);
        }
    }
    final private ByteBufferPool globalPool;
    final private int maxCachedPerThread;
    final private int lowestBufferId;
    final private int largestBufferSize;
    final private ThreadLocalCache cache;

    ThreadLocalByteBufferPool(ByteBufferPool globalPool) {
        this(globalPool, 0x100000, 1024, 0x40000);
    }
    ThreadLocalByteBufferPool(ByteBufferPool globalPool, int maxCachedPerThread, int smallestBufferSize, int largestBufferSize) {
        this.globalPool = globalPool;
        this.maxCachedPerThread = maxCachedPerThread;
        this.lowestBufferId = bufferId(smallestBufferSize);
        this.largestBufferSize = largestBufferSize;
        cache =  new ThreadLocalCache(bufferId(largestBufferSize));
    }
    @Override
    public ByteBuffer acquire(int size, boolean direct) {
        if (size <= largestBufferSize) {
            ByteBuffer buf = cache.get().acquire(Integer.min(lowestBufferId, bufferId(size)), direct);
            if (buf != null) return buf;
        }
        return globalPool.acquire(size, direct);
    }

    @Override
    public void release(ByteBuffer buffer) {
        if (buffer.capacity() <= largestBufferSize) {
            Cache local = cache.get();
            if (local.cachedBytes < maxCachedPerThread) {
                buffer = local.release(Integer.max(lowestBufferId, bufferId(buffer.capacity())), buffer);
            }
        }
        if (buffer != null) {
            globalPool.release(buffer);
        }
    }
}
