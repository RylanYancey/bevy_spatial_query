use std::{hint, ops::Range};

use aligned_vec::{avec, AVec};
use bevy::{ecs::entity::Entity, math::IVec3};


pub struct EntityTree<const W: usize> {
    alloc: Vec<Bucket<W>>,
    x_values: AVec<i32>,
    buckets: Vec<Bucket<W>>,
}

impl<const W: usize> EntityTree<W> {
    pub fn new() -> Self {
        Self {
            alloc: Vec::new(),
            x_values: avec![],
            buckets: Vec::new(),
        }
    }

    fn alloc(&mut self) -> Bucket<W> {
        self.alloc.pop().unwrap_or_else(|| Bucket { items: Vec::new() })
    }

    pub fn clear(&mut self) {
        self.x_values.clear();
        while let Some(mut bucket) = self.buckets.pop() {
            bucket.items.clear();
            self.alloc.push(bucket);
        }
    }

    fn x_range(&self, min_x: i32, max_x: i32) -> Range<usize> {
        if self.x_values.last().is_none_or(|&x| x < min_x) || self.x_values[0] > max_x {
            return 0..0
        }

        let mut size = self.x_values.len();
        let mut lower = 0usize;
        let mut upper = size;

        while size > 1 {
            let half = size / 2;
            let mid = lower + half;
            let cmp = self.x_values[mid] <= min_x;
            lower = hint::select_unpredictable(cmp, mid, lower);
            let mid = upper - half;
            let cmp = self.x_values[mid] >= max_x;
            upper = hint::select_unpredictable(cmp, mid, upper); 
            size -= half;
        }

        lower..upper
    }

    pub fn push(&mut self, entity: Entity, point: IVec3) {
        let x = point.x & !(W - 1) as i32;
        match self.x_values.binary_search(&x) {
            Ok(i) => self.buckets[i].add(entity, point),
            Err(i) => {
                self.x_values.insert(i, x);
                let mut bucket = self.alloc();
                bucket.add(entity, point);
                self.buckets.insert(i, bucket);
            }
        }
    }

    pub fn build(&mut self) {
        for bucket in &mut self.buckets {
            bucket.sort();
        }
    }

    pub fn in_range<'t>(&'t self, point: IVec3, range: i32) -> InRange<'t, W> {
        let point = point & !(W - 1) as i32;
        let range = (range + (W as i32 - 1)) & !(W - 1) as i32;
        let min = point - range;
        let max = point + range;
        InRange {
            tree: self,
            x_range: self.x_range(min.x, max.x),
            z_range: 0..0,
            x_curr: 0,
            min,
            max
        }
    }

    pub fn in_range_xz<'t>(&'t self, point: IVec3, range: i32) -> InRangeXz<'t, W> {
        let point = point & !(W - 1) as i32;
        let range = (range + W as i32 - 1) & !(W - 1) as i32;
        let min = point - range;
        let max = point + range;
        InRangeXz {
            tree: self,
            x_range: self.x_range(min.x, max.x),
            z_range: 0..0,
            x_curr: 0,
            min,
            max
        }
    }
}

struct Item {
    entity: Entity,
    y: i32,
    z: i32,
}

struct Bucket<const W: usize> {
    items: Vec<Item>,
}

impl<const W: usize> Bucket<W> {
    fn add(&mut self, entity: Entity, point: IVec3) {
        self.items.push(Item { 
            entity, 
            y: point.y, 
            z: point.z, 
        })
    }

    fn sort(&mut self) {
        // sort by z, then by y.
        self.items.sort_unstable_by(|a, b| {
            if a.z != b.z { a.z.cmp(&b.z) }
            else { a.y.cmp(&b.y) }
        });
    }

    /// Range of entities in the bucket that have a z value in-range.
    fn z_range(&self, min_z: i32, max_z: i32) -> Range<usize> {
        if self.items.last().is_none_or(|item| item.z < min_z) || self.items[0].z > max_z {
            return 0..0
        }

        let mut size = self.items.len();
        let mut lower = 0usize;
        let mut upper = size;

        while size > 1 {
            let half = size / 2;
            let mid = lower + half;
            let cmp = self.items[mid].z <= min_z;
            lower = hint::select_unpredictable(cmp, mid, lower);
            let mid = upper - half;
            let cmp = self.items[mid].z >= max_z;
            upper = hint::select_unpredictable(cmp, mid, upper); 
            size -= half;
        }

        lower..upper
    }
}

pub struct InRange<'t, const W: usize> {
    tree: &'t EntityTree<W>,
    x_range: Range<usize>,
    z_range: Range<usize>,
    x_curr: usize,
    min: IVec3,
    max: IVec3,
}

impl<'t, const W: usize> Iterator for InRange<'t, W> {
    type Item = Entity;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(i) = self.z_range.next() {
                let item = &self.tree.buckets[self.x_curr].items[i];
                if item.y >= self.min.y && item.y < self.max.y {
                    return Some(item.entity)
                } 
            } else {
                if let Some(i) = self.x_range.next() {
                    self.z_range = self.tree.buckets[i].z_range(self.min.z, self.max.z);
                    self.x_curr = i;
                } else {
                    return None;
                }
            }
        }
    }
}

pub struct InRangeXz<'t, const W: usize> {
    tree: &'t EntityTree<W>,
    x_range: Range<usize>,
    z_range: Range<usize>,
    x_curr: usize,
    min: IVec3,
    max: IVec3,
}

impl<'t, const W: usize> Iterator for InRangeXz<'t, W> {
    type Item = Entity;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(i) = self.z_range.next() {
                return Some(self.tree.buckets[self.x_curr].items[i].entity)
            } else {
                if let Some(i) = self.x_range.next() {
                    self.z_range = self.tree.buckets[i].z_range(self.min.z, self.max.z);
                    self.x_curr = i;
                } else {
                    return None;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bevy::{ecs::entity::Entity, math::IVec3};

    use super::EntityTree;

    struct TestRng(pub u64);

    impl TestRng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(39756357833399321);
            let r = u128::from(self.0).wrapping_mul(937373313885393111343);
            ((r >> 64) ^ r) as u64
        }
    }

    #[test]
    fn test_in_range_z_only() {
        let mut tree = EntityTree::<32>::new();
        let mut entities = Vec::new();
        let mut points = Vec::new();
        for i in 0..100 {
            let point = IVec3::new(0, 0, i);
            let entity = Entity::from_raw(i as u32);
            entities.push(entity);
            points.push(point);
            tree.push(entity, point);
        }
        tree.build();

        let pt = IVec3::new(0, 0, 45);
        let range = 29;

        let iter = tree.in_range(pt, range);

        let mut in_range = HashSet::new();
        for i in 0..points.len() {
            if points[i].z >= iter.min.z &&
                points[i].z < iter.max.z 
            {
                in_range.insert(entities[i]);
            }
        }

        for entity in iter {
            assert!(in_range.remove(&entity), "Failed on: {:?}", points[entity.index() as usize]);
        }

        assert_eq!(in_range.len(), 0);
    }

    #[test]
    fn test_in_range_x_only() {
        let mut tree = EntityTree::<32>::new();
        let mut entities = Vec::new();
        let mut points = Vec::new();
        for i in 0..100 {
            let point = IVec3::new(i, 0, 0);
            let entity = Entity::from_raw(i as u32);
            entities.push(entity);
            points.push(point);
            tree.push(entity, point);
        }
        tree.build();

        let pt = IVec3::new(45, 0, 0);
        let range = 29;

        let iter = tree.in_range(pt, range);

        let mut in_range = HashSet::new();
        for i in 0..points.len() {
            if points[i].x >= iter.min.x &&
                points[i].x < iter.max.x 
            {
                in_range.insert(entities[i]);
            }
        }

        for entity in iter {
            assert!(in_range.remove(&entity), "Failed on: {:?}", points[entity.index() as usize]);
        }

        assert_eq!(in_range.len(), 0);
    }

    #[test]
    fn test_in_range() {
        let mut rng = TestRng(3985637837831);
        let mut tree = EntityTree::<32>::new();
        let mut points = Vec::new();
        let mut entities = Vec::new();
        for i in 0..1000 {
            let x = (rng.next() & 511) as i32 - 256;
            let y = (rng.next() & 511) as i32 - 256;
            let z = (rng.next() & 511) as i32 - 256;
            let point = IVec3::new(x, y, z);
            let entity = Entity::from_raw(i);
            points.push(point);
            entities.push(entity);
            tree.push(entity, point);
        }
        tree.build();

        let pt = IVec3::new(55, 69, -72);
        let range = 139;

        let iter = tree.in_range(pt, range);

        let mut in_range = HashSet::new();
        for i in 0..points.len() {
            let IVec3 { x, y, z } = points[i]; 
            if x >= iter.min.x && x < iter.max.x 
                && y >= iter.min.y && y < iter.max.y
                && z >= iter.min.z && z < iter.max.z
            {
                in_range.insert(entities[i]);
            }
        }

        for entity in iter {
            assert!(in_range.remove(&entity), "Failed on: {:?}", points[entity.index() as usize]);
        }

        assert_eq!(in_range.len(), 0);
    }
}