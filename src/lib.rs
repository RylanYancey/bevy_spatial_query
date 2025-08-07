
#![feature(select_unpredictable)]

use std::{marker::PhantomData, ops::Deref, time::Duration};
use bevy::{prelude::*, time::common_conditions::on_timer};
use crate::tree::EntityTree;

const B: usize = 32;

pub mod tree;

#[derive(Resource)]
pub struct SpatialQuery<T> {
    tree: EntityTree<B>,
    _marker: PhantomData<T>
}

impl<T> Deref for SpatialQuery<T> {
    type Target = EntityTree<B>;

    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}

impl<T> Default for SpatialQuery<T> {
    fn default() -> Self {
        Self {
            tree: EntityTree::new(),
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
        spatial.tree.push(entity, transform.translation().as_ivec3());
    }
    spatial.tree.build();
}

