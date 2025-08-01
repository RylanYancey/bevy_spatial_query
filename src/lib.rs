
#![feature(select_unpredictable)]
#![feature(portable_simd)]

//! This crate provides a `SpatialQuery` resource that allows searching for entities within range of a point.
//! 
//! The implementation is heavily based on Kiddo's ImmutableKdTree, but not exactly. While we support querying in a 3d bounds,
//! no actual partitioning is done in the Y dimension, only in the X and Z dimensions. This improves query time IF your entities
//! are on roughly the same Y-level. For Opencraft, which this crate was built for, that is the case. If your game is 2D or has 
//! a very far apart upper and lower bounds, this crate probably won't work well for you.
//! 
//! ## Features
//!  - 2D/3D Euclidian Distance queries
//!  - 2D/3D Volume/Area queries
//!  - KNN Search (Needs optimization, but usable)
//!  - Uses iterators instead of collecting into a Vector every time
//!
//! ## Usage
//! 
//! ```
//! use bevy::prelude::*;
//! use bevy_spatial_query::{AppSpatialQueryExt, SpatialQueryPlugin, SpatialQuery};
//! 
//! #[derive(Component)]
//! struct Tracked;
//! 
//! fn main() {
//!     App::new()
//!         .add_plugins((
//!             DefaultPlugins,
//!             SpatialQueryPlugin {
//!                 // rebuild 10 times per second
//!                 rebuild_frequency: 10,
//!             }
//!         ))
//!         .init_spatial_tracking::<Tracked>() // < Loads resources and rebuilding
//!         .add_systems(
//!             Update,
//!             query_entities_in_range, // < Use with Res<SpatialQuery<Tracked>>
//!         );
//! }
//! 
//! fn query_entities_in_range(
//!     query: Res<SpatialQuery<Tracked>>,
//! ) {
//!     for nearby in query.in_circle(Vec3::new(10.0, 4.0, -2.0), 5.0) {
//!         // ...do something with entity and distance
//!     }
//! }
//! ```

use std::{marker::PhantomData, ops::Deref, time::Duration};

use bevy::{prelude::*, time::common_conditions::on_timer};

use crate::tree::PartitionTree;

pub mod tree;

#[derive(Resource)]
pub struct SpatialQuery<T> {
    tree: PartitionTree,
    _marker: PhantomData<T>
}

impl<T> Deref for SpatialQuery<T> {
    type Target = PartitionTree;

    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}

impl<T> Default for SpatialQuery<T> {
    fn default() -> Self {
        Self {
            tree: PartitionTree::new(),
            _marker: PhantomData,
        }
    }
}

pub struct SpatialQueryPlugin {
    /// Max times per second the tree is rebuilt. 
    pub rebuild_frequency: u32,
}

impl Plugin for SpatialQueryPlugin {
    fn build(&self, app: &mut App) {
        app
            .configure_sets(
                Update,
                SpatialQueryRebuildSet
                    .run_if(on_timer(
                        Duration::from_secs_f32(1.0 / self.rebuild_frequency as f32)
                    ))
            )
        ;
    }
}

#[derive(SystemSet, Clone, Eq, PartialEq, Debug, Hash)]
pub struct SpatialQueryRebuildSet;

pub trait AppSpatialQueryExt {
    fn init_spatial_tracking<T>(&mut self) -> &mut Self
    where
        T: Component;
}

impl AppSpatialQueryExt for App {
    fn init_spatial_tracking<T>(&mut self) -> &mut Self 
    where
        T: Component
    {
        self.init_resource::<SpatialQuery<T>>()
            .add_systems(
                Update,
                update_partition_tree::<T>
                    .in_set(SpatialQueryRebuildSet)
            )
    }
}

fn update_partition_tree<T>(
    query: Query<(&GlobalTransform, Entity), With<T>>,
    mut spatial: ResMut<SpatialQuery<T>>,
)
where
    T: Component
{
    spatial.tree.clear();
    for (transform, entity) in &query {
        spatial.tree.push(transform.translation(), entity);
    }
    spatial.tree.rebuild_in_place();
}