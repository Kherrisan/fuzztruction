use std::hash::Hash;

use hashlink::LinkedHashSet;

/// 一个简单的有界集合，基于 `LinkedHashSet` 保留插入顺序。
/// 当元素数量超过容量时，会自动移除最早插入的元素。
#[derive(Clone, Debug)]
pub struct BoundedSet<T>
where
    T: Eq + Hash + Default,
{
    inner: LinkedHashSet<T>,
    capacity: usize,
}

impl<T> BoundedSet<T>
where
    T: Eq + Hash + Default,
{
    /// 创建一个指定容量的有界集合。
    /// `capacity` 必须大于 0。
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            inner: LinkedHashSet::with_capacity(capacity),
            capacity,
        }
    }

    /// 返回当前容量上限。
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 返回当前元素数量。
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// 是否包含元素。
    pub fn contains(&self, value: &T) -> bool {
        self.inner.contains(value)
    }

    /// 尝试插入元素。
    ///
    /// 返回 `(inserted, evicted)`：
    /// - `inserted`：如果集合中原本不存在该元素则为 `true`。
    /// - `evicted`：如果超出容量而移除的最早插入元素。
    ///
    /// 若元素已存在，会将其移动到最新位置，不触发淘汰。
    pub fn insert(&mut self, value: T) -> (bool, Option<T>) {
        if self.inner.contains(&value) {
            // 重新插入以更新顺序到队尾
            self.inner.remove(&value);
            self.inner.insert(value);
            return (false, None);
        }

        let inserted = self.inner.insert(value);
        let evicted = if self.inner.len() > self.capacity {
            self.inner.pop_front()
        } else {
            None
        };
        (inserted, evicted)
    }

    /// 移除指定元素。
    pub fn remove(&mut self, value: &T) -> bool {
        self.inner.remove(value)
    }

    /// 清空集合。
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// 迭代集合中的元素（按插入顺序）。
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter()
    }
}
