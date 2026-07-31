import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict
import numpy as np


class DiskCache:
    """
    Disk-based LRU cache for depth maps.
    Automatically manages cache size and evicts old entries.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/depth_maps",
        max_size_gb: float = 10.0,
        enable_compression: bool = True,
    ):
        """
        Args:
            cache_dir: Directory to store cached depth maps
            max_size_gb: Maximum cache size in GB (default: 10GB)
            enable_compression: Use np.savez_compressed (slower but smaller)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.enable_compression = enable_compression

        # LRU tracking (most recent = end of dict)
        self.lru_tracker: OrderedDict[str, int] = OrderedDict()
        self.current_size_bytes = 0

        # Thread safety
        self.lock = threading.Lock()

        # Load existing cache metadata
        self._scan_cache()

    def _scan_cache(self) -> None:
        """Scan cache directory and build LRU tracker."""
        for file_path in self.cache_dir.glob("*.npz"):
            key = file_path.stem
            file_size = file_path.stat().st_size
            self.lru_tracker[key] = file_size
            self.current_size_bytes += file_size

    def _hash_key(self, key: str) -> str:
        """Generate filename-safe hash from key."""
        return hashlib.md5(key.encode()).hexdigest()

    def _evict_lru(self, required_bytes: int) -> None:
        """Evict least recently used items until space is available."""
        while (
            self.current_size_bytes + required_bytes > self.max_size_bytes
            and len(self.lru_tracker) > 0
        ):
            # Remove oldest item (first in OrderedDict)
            oldest_key, file_size = self.lru_tracker.popitem(last=False)

            cache_file = self.cache_dir / f"{oldest_key}.npz"
            if cache_file.exists():
                cache_file.unlink()
                self.current_size_bytes -= file_size
                print(f"[Cache] Evicted: {oldest_key} ({file_size / 1024**2:.1f} MB)")

    def put(
        self, key: str, depth_map: np.ndarray, metadata: Optional[Dict] = None
    ) -> None:
        """
        Store depth map to disk cache.

        Args:
            key: Unique identifier (e.g., image filename or hash)
            depth_map: Depth map array to cache
            metadata: Optional dict with additional info (timestamp, resolution, etc.)
        """
        with self.lock:
            hashed_key = self._hash_key(key)
            cache_file = self.cache_dir / f"{hashed_key}.npz"

            # Save to disk
            if self.enable_compression:
                np.savez_compressed(
                    cache_file, depth=depth_map, metadata=metadata or {}
                )
            else:
                np.savez(cache_file, depth=depth_map, metadata=metadata or {})

            file_size = cache_file.stat().st_size

            # Update LRU tracker
            if hashed_key in self.lru_tracker:
                # Remove old entry size
                self.current_size_bytes -= self.lru_tracker[hashed_key]
                del self.lru_tracker[hashed_key]

            # Add new entry (moves to end = most recent)
            self.lru_tracker[hashed_key] = file_size
            self.current_size_bytes += file_size

            # Evict if over size limit
            self._evict_lru(0)

            print(f"[Cache] Stored: {key} ({file_size / 1024**2:.1f} MB)")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve depth map from cache.

        Args:
            key: Unique identifier

        Returns:
            {"depth": np.ndarray, "metadata": dict} or None if not found
        """
        with self.lock:
            hashed_key = self._hash_key(key)
            cache_file = self.cache_dir / f"{hashed_key}.npz"

            if not cache_file.exists():
                return None

            # Load from disk
            data = np.load(cache_file, allow_pickle=True)

            # Update LRU (move to end = most recent)
            if hashed_key in self.lru_tracker:
                self.lru_tracker.move_to_end(hashed_key)

            print(f"[Cache] Retrieved: {key}")

            return {
                "depth": data["depth"],
                "metadata": data["metadata"].item() if "metadata" in data else {},
            }

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self.lock:
            hashed_key = self._hash_key(key)
            return (self.cache_dir / f"{hashed_key}.npz").exists()

    def clear(self) -> None:
        """Clear entire cache."""
        with self.lock:
            for cache_file in self.cache_dir.glob("*.npz"):
                cache_file.unlink()

            self.lru_tracker.clear()
            self.current_size_bytes = 0
            print("[Cache] Cleared all entries")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "num_entries": len(self.lru_tracker),
                "size_mb": self.current_size_bytes / 1024**2,
                "size_gb": self.current_size_bytes / 1024**3,
                "max_size_gb": self.max_size_bytes / 1024**3,
                "usage_percent": (self.current_size_bytes / self.max_size_bytes) * 100,
            }
