

use std::{cmp::Ordering, collections::BinaryHeap, hint, ops::Range, simd::{cmp::SimdPartialOrd, Simd}};

use aligned_vec::{avec, AVec};
use bevy::math::{Vec2, Vec3, Vec3Swizzles};
use bevy::ecs::entity::Entity;

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
const LANES: usize = 4;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
const LANES: usize = 8;

#[cfg(target_feature = "avx512f")]
const LANES: usize = 16;

#[allow(non_camel_case_types)]
type f32xL = Simd<f32, LANES>;

/// Bucket width.
/// Each Z bucket is B elements wide.
/// Each X bucket is B^2 elements wide, or B Z Buckets wide.
pub const B: usize = 16;

pub struct PartitionTree {
    x_axis: Vec<f32>,
    y_axis: Vec<f32>,
    z_axis: Vec<f32>,
    points: Vec<Vec3>,
    entities: Vec<Entity>,
    temp: Vec<Entity>,
    x_buckets: AVec<f32>,
    z_buckets: AVec<f32>,
    sort_index: Vec<usize>,
}

impl PartitionTree {
    pub fn new() -> Self {
        Self {
            x_axis: Vec::new(),
            y_axis: Vec::new(),
            z_axis: Vec::new(),
            points: Vec::new(),
            entities: Vec::new(),
            temp: Vec::new(),
            x_buckets: avec![],
            z_buckets: avec![],
            sort_index: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.x_axis.clear();
        self.z_axis.clear();
        self.y_axis.clear();
        self.points.clear();
        self.entities.clear();
        self.temp.clear();
        self.x_buckets.clear();
        self.z_buckets.clear();
        self.sort_index.clear();
    }

    /// Appends to the end of the Tree's buffer. 
    /// Tree lookups are meaningless until rebuild_in_place is called.
    pub fn push(&mut self, pos: Vec3, entity: Entity) {
        self.points.push(pos);
        self.temp.push(entity)
    }

    /// The number of entities in the Tree.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn rebuild_in_place(&mut self) {
        self.sort_index.clear();
        for i in 0..self.len() {
            self.sort_index.push(i);
        }

        // Sort X partitions by X
        for SubDivision { range, pivot } in SubDivisions::new(self.len(), B*B) {
            // Sort such that all X values above the pivot are greater than the pivot, 
            // and all X values below the pivot are less than the pivot. 
            self.sort_index[range].select_nth_unstable_by(pivot, 
                |&a, &b| self.points[a].x.partial_cmp(&self.points[b].x).unwrap());
        }

        // Sort Z partitions by Z
        for chunk in Chunks::new(self.len(), B*B) {
            for SubDivision { range, pivot } in SubDivisions::new(chunk.len(), B) {
                // Sort such that, within this Z subdivision of the X chunk, all Z values above the 
                // pivot are greater than the pivot, and all Z values below the pivot are less than the pivot.
                self.sort_index[range].select_nth_unstable_by(pivot,
                    |&a, &b| self.points[a].z.partial_cmp(&self.points[b].z).unwrap());
            }

            // Compute minimum x value in each x partition
            self.x_buckets.push(
                self.sort_index[chunk]
                    .iter()
                    .map(|&i| self.x_axis[i])
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(f32::NEG_INFINITY)
            );
        }

        // compute minimum z in each z chunk
        for chunk in Chunks::new(self.len(), B) {
            self.z_buckets.push(
                self.sort_index[chunk]
                    .iter()
                    .map(|&i| self.z_axis[i])
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(f32::NEG_INFINITY)
            );
        }

        // Move unsorted points and entities into their final destination.
        for i in 0..self.len() {
            let j = self.sort_index[i];
            self.x_axis[i] = self.points[j].x;
            self.y_axis[i] = self.points[j].y;
            self.z_axis[i] = self.points[j].z;
            self.entities[i] = self.temp[j];
        }

        // round up to next multiple of Lanes so we don't have to handle chunk remainders.
        let new_size = self.x_axis.len().next_multiple_of(LANES);
        self.x_axis.resize(new_size, f32::INFINITY);
        self.y_axis.resize(new_size, f32::INFINITY);
        self.z_axis.resize(new_size, f32::INFINITY);
        self.entities.resize(new_size, Entity::PLACEHOLDER);
    }

    /// Get the last x partition where the min value is less than min_x.
    fn first_relevant_x_partition(&self, min_x: f32) -> usize {
        match self.x_buckets.binary_search_by(|&x| {
            if x < min_x {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1)
        }
    }

    fn relevant_z_in_x(&self, x: usize, min_z: f32, max_z: f32) -> Range<usize> {
        use std::cmp::Ordering::*;
        let z = x * B;
        let vals = &self.z_buckets[z..(z+B).min(self.z_buckets.len())];
        let mut size = vals.len();
        let mut min = 0usize;
        let mut max = 0usize;

        // ripped from the implementation of Vec::binary_search_by
        while size > 1 {
            let half = size / 2;
            let mid = min + half;
            let cmp = if vals[mid] < min_z { Less } else { Greater };
            min = hint::select_unpredictable(cmp == Greater, min, mid);
            let mid = max + half;
            let cmp = if vals[mid] < max_z { Less } else { Greater };
            max = hint::select_unpredictable(cmp == Greater, max, mid);
            size -= half
        }

        min..max+1
    }

    /// Iter all nearby entities within the shape.
    pub fn within<'t, S: QueryShape>(&'t self, point: Vec3, radius: f32) -> Within<'t, S> {
        Within::new(self, point, radius)
    }

    /// Iter all nearby entities within a 3D Sphere, given a center and a radius.
    pub fn in_sphere<'t>(&'t self, point: Vec3, radius: f32) -> Within<'t, Sphere> {
        self.within(point, radius)
    }

    /// Iter all nearby entities within a 2D Circle, given a center and a radius.
    pub fn in_circle<'t>(&'t self, point: Vec3, radius: f32) -> Within<'t, Circle> {
        self.within(point, radius)
    }

    /// Iter all nearby entities within a 2D Area, given a center and a half-extent.
    pub fn in_area<'t>(&'t self, point: Vec3, half_extent: f32) -> Within<'t, Area> {
        self.within(point, half_extent)
    }

    /// Iter all nearby entities in a 3D Volume, given a center and a half-extent.
    pub fn in_volume<'t>(&'t self, point: Vec3, half_extent: f32) -> Within<'t, Volume> {
        self.within(point, half_extent)
    }
}

pub struct Within<'t, S> {
    shape: S,
    iter: Partitions<'t>,
    curr: Partition<'t>,
}

impl<'t, S> Within<'t, S> 
where
    S: QueryShape,
{
    pub fn new(tree: &'t PartitionTree, pt: Vec3, radius: f32) -> Self {
        let mut iter = Partitions::new(tree, pt.xz(), radius);
        let range = iter.next().unwrap_or(0..0);
        Self {
            shape: S::new(pt, radius),
            iter,
            curr: Partition::new(tree, range),
        }
    }
}

impl<'t, S> Within<'t, S>
where
    S: QueryShape<Output=Nearby>,
{
    /// Find the closest entity.
    pub fn nearest(self) -> Option<Nearby> {
        self.min()
    }

    fn nearest_n_heap(self, n: usize) -> BinaryHeap<Nearby> {
        let mut heap = BinaryHeap::new();
        for nearby in self {
            if heap.len() < n {
                heap.push(nearby);
            } else {
                let mut max_val = heap.peek_mut().unwrap();
                if nearby.distance < max_val.distance {
                    *max_val = nearby;
                } 
            }
        }
        heap
    }

    /// Find the N nearest entities, unsorted.
    pub fn nearest_n(self, n: usize) -> Vec<Nearby> {
        self.nearest_n_heap(n).into_vec()
    }

    /// Find and sort the nearest N entities.
    pub fn nearest_n_sorted(self, n: usize) -> Vec<Nearby> {
        self.nearest_n_heap(n).into_sorted_vec()
    }
}

impl<'t, S> Iterator for Within<'t, S>
where
    S: QueryShape
{
    type Item = S::Output;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.shape.next(&self.curr) {
                return Some(item)
            }

            if let Some(range) = self.iter.next() {
                self.curr = Partition::new(self.iter.tree, range);
                continue;
            }

            return None;
        }
    }
}

#[derive(Copy, Clone)]
pub struct Nearby {
    /// The Distance Squared
    pub distance: f32,

    /// The entity at this distance.
    pub entity: Entity,    
}

impl PartialOrd for Nearby {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Nearby {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Eq for Nearby {}
impl PartialEq for Nearby {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

#[derive(Copy, Clone)]
pub struct Partitions<'t> {
    tree: &'t PartitionTree,
    curr: usize,
    min: Vec2,
    max: Vec2,
}

impl<'t> Partitions<'t> {
    pub fn new(tree: &'t PartitionTree, point: Vec2, range: f32) -> Self {
        let min = point - Vec2::splat(range) / 2.0;
        let max = point + Vec2::splat(range) / 2.0;
        Partitions {
            tree, min, max,
            curr: tree.first_relevant_x_partition(min.x),
        }
    }
}

impl<'t> Iterator for Partitions<'t> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tree.x_buckets.get(self.curr).is_none_or(|&x| x > self.max.x) {
            None
        } else {
            let range = self.tree.relevant_z_in_x(self.curr, self.min.y, self.max.y);
            let result = Range { start: range.start * B, end: (range.end + 1) * B };
            self.curr += 1;
            Some(result)
        }
    }
}

pub struct Partition<'t> {
    x_axis: &'t [f32],
    y_axis: &'t [f32],
    z_axis: &'t [f32],
    entity: &'t [Entity],
}

impl<'t> Partition<'t> {
    pub fn new(tree: &'t PartitionTree, range: Range<usize>) -> Self {
        Self {
            x_axis: &tree.x_axis[range.clone()],
            y_axis: &tree.y_axis[range.clone()],
            z_axis: &tree.z_axis[range.clone()],
            entity: &tree.entities[range],
        }
    }

    pub fn len(&self) -> usize {
        self.x_axis.len()
    }
}

pub trait QueryShape {
    type Output;

    fn new(point: Vec3, radius: f32) -> Self;
    fn next(&mut self, partition: &Partition) -> Option<Self::Output>;
}

pub struct Circle {
    point: Vec2,
    rad_sq: f32,
    lanes: f32xL,
    curr: usize,
    mask: u32,
}

impl QueryShape for Circle {
    type Output = Nearby;

    fn new(point: Vec3, radius: f32) -> Self {
        Self { 
            point: point.xz(), rad_sq: radius * radius, 
            lanes: f32xL::splat(0.0), curr: 0, mask: 0,
        }
    }

    fn next(&mut self, partition: &Partition) -> Option<Self::Output> {
        while self.mask == 0 {
            self.curr += LANES;
            if self.curr >= partition.len() {
                self.curr = 0;
                return None
            } else {
                // x axis
                let sx = Simd::from_slice(&partition.x_axis[self.curr..]);
                let sx = sx - Simd::splat(self.point.x);
                let sx = sx * sx;
                // z axis
                let sz = Simd::from_slice(&partition.z_axis[self.curr..]);
                let sz = sz - Simd::splat(self.point.y);
                let sz = sz * sz;
                // combine
                self.lanes = sx + sz;
                // mask 
                self.mask = self.lanes.simd_le(Simd::splat(self.rad_sq)).to_bitmask() as u32;
            }
        }
        
        let j = self.mask.trailing_zeros();
        self.mask ^= 1 << j;
        Some(Nearby {
            distance: self.lanes[j as usize],
            entity: partition.entity[self.curr + j as usize]
        })
    }
}

pub struct Sphere {
    point: Vec3,
    rad_sq: f32,
    lanes: f32xL,
    curr: usize,
    mask: u32,
}

impl QueryShape for Sphere {
    type Output = Nearby;

    fn new(point: Vec3, radius: f32) -> Self {
        Self { 
            point, rad_sq: radius * radius, 
            lanes: f32xL::splat(0.0), curr: 0, mask: 0,
        }
    }

    fn next(&mut self, partition: &Partition) -> Option<Self::Output> {
        while self.mask == 0 {
            self.curr += LANES;
            if self.curr >= partition.len() {
                self.curr = 0;
                return None
            } else {
                // x axis
                let sx = Simd::from_slice(&partition.x_axis[self.curr..]);
                let sx = sx - Simd::splat(self.point.x);
                let sx = sx * sx;
                // y axis
                let sy = Simd::from_slice(&partition.y_axis[self.curr..]);
                let sy = sy - Simd::splat(self.point.y);
                let sy = sy * sy;
                // z axis
                let sz = Simd::from_slice(&partition.z_axis[self.curr..]);
                let sz = sz - Simd::splat(self.point.z);
                let sz = sz * sz;
                // combine
                self.lanes = sx + sy + sz;
                // mask 
                self.mask = self.lanes.simd_lt(Simd::splat(self.rad_sq)).to_bitmask() as u32;
            }
        }
        
        let j = self.mask.trailing_zeros();
        self.mask ^= 1 << j;
        Some(Nearby {
            distance: self.lanes[j as usize],
            entity: partition.entity[self.curr + j as usize]
        })
    }
}

pub struct Area {
    point: Vec2,
    range: f32,
    curr: usize,
    mask: u32,
}

impl QueryShape for Area {
    type Output = Entity;

    fn new(point: Vec3, radius: f32) -> Self {
        Self {
            point: point.xz(), 
            range: radius, 
            curr: 0, mask: 0
        }
    }

    fn next(&mut self, partition: &Partition) -> Option<Self::Output> {
        while self.mask == 0 {
            self.curr += LANES;
            if self.curr >= partition.len() {
                self.curr = 0;
                return None;
            } else {
                // x axis
                let vx: Simd<f32, LANES> = Simd::from_slice(&partition.x_axis[self.curr..]);
                let mx = vx.simd_ge(Simd::splat(self.point.x - self.range)) & vx.simd_le(Simd::splat(self.point.x + self.range));
                // z axis
                let vz: Simd<f32, LANES> = Simd::from_slice(&partition.z_axis[self.curr..]);
                let mz = vz.simd_ge(Simd::splat(self.point.y - self.range)) & vz.simd_le(Simd::splat(self.point.y + self.range));
                self.mask = (mx & mz).to_bitmask() as u32;
            }
        }

        let j = self.mask.trailing_zeros();
        self.mask ^= 1 << j;
        Some(partition.entity[self.curr + j as usize])
    }
}

pub struct Volume {
    point: Vec3,
    range: f32,
    curr: usize,
    mask: u32,
}

impl QueryShape for Volume {
    type Output = Entity;

    fn new(point: Vec3, radius: f32) -> Self {
        Self {
            point, range: radius, 
            curr: 0, mask: 0
        }
    }

    fn next(&mut self, partition: &Partition) -> Option<Self::Output> {
        while self.mask == 0 {
            self.curr += LANES;
            if self.curr >= partition.len() {
                self.curr = 0;
                return None;
            } else {
                // x axis
                let vx: f32xL = Simd::from_slice(&partition.x_axis[self.curr..]);
                let mx = vx.simd_ge(Simd::splat(self.point.x - self.range)) & vx.simd_le(Simd::splat(self.point.x + self.range));
                // y axis
                let vy: f32xL = Simd::from_slice(&partition.y_axis[self.curr..]);
                let my = vy.simd_ge(Simd::splat(self.point.y - self.range)) & vy.simd_le(Simd::splat(self.point.y + self.range));
                // z axis
                let vz: f32xL = Simd::from_slice(&partition.z_axis[self.curr..]);
                let mz = vz.simd_ge(Simd::splat(self.point.z - self.range)) & vz.simd_le(Simd::splat(self.point.z + self.range));
                self.mask = (mx & my & mz).to_bitmask() as u32;
            }
        }

        let j = self.mask.trailing_zeros();
        self.mask ^= 1 << j;
        Some(partition.entity[self.curr + j as usize])
    }
}

/// Visit subdivisions of a slice. 
/// The pattern is like this:
///  - 0..512
///  - 0..256,256..512
///  - 0..128,128..256,256..384,384..512
/// 
/// This iterator will round the length up to the
/// nearest power of two, so an input length of 257
/// will round up to 512. Any partitions where the center point
/// is out-of-bounds will be ignored.
#[derive(Copy, Clone)]
pub struct SubDivisions {
    /// The subdivision width to stop at.
    /// Any subdivisions this size or smaller will cause the iterator to exit.
    stop_at: usize,
    /// The current subdivision width.
    curr_size: usize,
    /// The current subdivision index.
    curr_idx: usize,
    /// Length of the slice.
    len: usize,
}

impl SubDivisions {
    pub fn new(len: usize, limit: usize) -> Self {
        Self {
            stop_at: limit,
            len,
            curr_size: len.next_power_of_two(),
            curr_idx: 0,
        }
    }
}

impl Iterator for SubDivisions {
    type Item = SubDivision;

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr_size > self.stop_at {
            let result = SubDivision {
                range: self.curr_idx..self.curr_idx + self.curr_size,
                pivot: self.curr_size >> 1,
            };

            if result.pivot > self.len {
                self.curr_size >>= 1;
                self.curr_idx = 0;
            } else {
                return Some(result)
            }
        }

        None
    }
}

#[derive(Clone)]
pub struct SubDivision {
    /// The range of values
    pub range: Range<usize>,

    /// The center point relative to the start of the range.
    /// Equivalent to `range.len() >> 1`
    pub pivot: usize,
}

/// A Very simple iterator over chunks in a length
pub struct Chunks {
    len: usize,
    curr: usize,
    size: usize,
}

impl Chunks {
    pub fn new(len: usize, size: usize) -> Self {
        Self { len, curr: 0, size }
    }
}

impl Iterator for Chunks {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr >= self.len {
            None 
        } else {
            let start = self.curr;
            self.curr += self.size;
            Some(start..self.curr.min(self.len))
        }
    }
}