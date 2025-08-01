# bevy_spatial_query

This crate provides a `SpatialQuery` resource that allows searching for entities within range of a point.

The implementation is heavily based on Kiddo's ImmutableKdTree, but not exactly. While we support querying in a 3d bounds,
no actual partitioning is done in the Y dimension, only in the X and Z dimensions. This improves query time IF your entities
are on roughly the same Y-level. For Opencraft, which this crate was built for, that is the case. If your game is 2D or has
a very far apart upper and lower bounds, this crate probably won't work well for you.

### Features
 - 2D/3D Euclidian Distance queries
 - 2D/3D Volume/Area queries
 - KNN Search (Needs optimization, but usable)
 - Uses iterators instead of collecting into a Vector every time

### Usage

```rust
use bevy::prelude::*;
use bevy_spatial_query::{AppSpatialQueryExt, SpatialQueryPlugin, SpatialQuery};

#[derive(Component)]
struct Tracked;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            SpatialQueryPlugin {
                // rebuild 10 times per second
                rebuild_frequency: 10,
            }
        ))
        .init_spatial_tracking::<Tracked>() // < Loads resources and rebuilding
        .add_systems(
            Update,
            query_entities_in_range, // < Use with Res<SpatialQuery<Tracked>>
        );
}

fn query_entities_in_range(
    query: Res<SpatialQuery<Tracked>>,
) {
    for nearby in query.in_circle(Vec3::new(10.0, 4.0, -2.0), 5.0) {
        // ...do something with entity and distance
    }
}
```
