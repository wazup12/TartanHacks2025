// src/App.js
import React, { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const App = () => {
  // Ref for the map container element.
  const mapContainerRef = useRef(null);
  // Ref to store the MapLibre map instance.
  const mapRef = useRef(null);
  // Ref to store all fire markers for later removal.
  const fireMarkersRef = useRef([]);
  // State to hold the search query.
  const [searchQuery, setSearchQuery] = useState('');

  // Replace with your valid MapTiler API key.
  const maptilerApiKey = 'cExLiUPwUmrQLc74o84k';

  useEffect(() => {
    if (!mapContainerRef.current) return; // Ensure the container exists

    // Define the style URL for MapTiler's hybrid (satellite imagery with labels) style.
    const styleURL = `https://api.maptiler.com/maps/hybrid/style.json?key=${maptilerApiKey}`;

    // Initialize the map.
    mapRef.current = new maplibregl.Map({
      container: mapContainerRef.current,
      style: styleURL,
      center: [-118.2437, 34.0522], // Centered on Los Angeles.
      zoom: 8,
      pitch: 60,    // Add pitch for a 3D look.
      bearing: -10, // Slight rotation.
    });

    // Add navigation controls (zoom and rotation buttons).
    mapRef.current.addControl(new maplibregl.NavigationControl(), 'top-right');

    // When the map has finished loading the style...
    mapRef.current.on('load', () => {
      // Add a DEM (Digital Elevation Model) source for 3D terrain.
      if (!mapRef.current.getSource('maptiler-dem')) {
        mapRef.current.addSource('maptiler-dem', {
          type: 'raster-dem',
          tiles: [
            `https://api.maptiler.com/tiles/terrain-rgb/{z}/{x}/{y}.png?key=${maptilerApiKey}`
          ],
          tileSize: 256,
          maxzoom: 14,
        });
      }

      // Enable 3D terrain with an exaggeration factor.
      mapRef.current.setTerrain({ source: 'maptiler-dem', exaggeration: 1.5 });

      // (Optional) Add a sky layer for atmospheric effects.
      mapRef.current.addLayer({
        id: 'sky',
        type: 'sky',
        paint: {
          'sky-type': 'atmosphere',
          'sky-atmosphere-sun': [0.0, 0.0],
          'sky-atmosphere-sun-intensity': 15,
        },
      });
    });

    // Set up the click handler to add a fire marker.
    mapRef.current.on('click', (event) => {
      const { lng, lat } = event.lngLat;
      console.log(`Map clicked at: ${lng}, ${lat}`);

      // Create a DOM element to serve as the fire marker.
      const fireEl = document.createElement('div');
      fireEl.style.width = '12px';
      fireEl.style.height = '12px';
      fireEl.style.backgroundColor = 'orange';
      fireEl.style.borderRadius = '50%';
      fireEl.style.boxShadow = '0 0 8px 3px rgba(255, 140, 0, 0.8)';

      // Create a marker with the element and add it to the map.
      const marker = new maplibregl.Marker({ element: fireEl, anchor: 'center' })
        .setLngLat([lng, lat])
        .addTo(mapRef.current);

      // Save the marker reference for later removal.
      fireMarkersRef.current.push(marker);
    });

    // Cleanup the map instance on unmount.
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
      }
    };
  }, [maptilerApiKey]);

  // Handler for submitting the search form.
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    try {
      // Use MapTiler's Geocoding API to search for the location.
      const response = await fetch(
        `https://api.maptiler.com/geocoding/${encodeURIComponent(searchQuery)}.json?key=${maptilerApiKey}`
      );
      if (!response.ok) {
        console.error('Geocoding request failed:', response.statusText);
        return;
      }
      const data = await response.json();
      console.log('Geocoding result:', data);

      if (data.features && data.features.length > 0) {
        // Use the first search result.
        const feature = data.features[0];
        const [lng, lat] = feature.center;
        // Fly the map to the searched location.
        mapRef.current.flyTo({
          center: [lng, lat],
          zoom: 12,
          essential: true,
        });
      } else {
        alert('No location found for your search.');
      }
    } catch (error) {
      console.error('Error during geocoding:', error);
    }
  };

  // Handler for the "Reset Fire" button.
  const handleResetFire = () => {
    fireMarkersRef.current.forEach((marker) => marker.remove());
    fireMarkersRef.current = [];
  };

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      {/* Map container */}
      <div
        ref={mapContainerRef}
        style={{ width: '100%', height: '100%' }}
      />

      {/* Overlay UI */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        backgroundColor: 'white',
        padding: '10px',
        borderRadius: '4px',
        boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
        zIndex: 1,
      }}>
        <form onSubmit={handleSearch} style={{ display: 'flex', marginBottom: '10px' }}>
          <input
            type="text"
            placeholder="Search for a location..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ flex: 1, marginRight: '5px', padding: '5px' }}
          />
          <button type="submit" style={{ padding: '5px 10px' }}>Search</button>
        </form>
        <button
          onClick={handleResetFire}
          style={{ padding: '5px 10px', width: '100%' }}
        >
          Reset Fire
        </button>
      </div>
    </div>
  );
};

export default App;
