// src/App.js
import React, { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const App = () => {
  // Reference for the map container element.
  const mapContainerRef = useRef(null);
  // Reference to store the MapLibre map instance.
  const mapRef = useRef(null);
  // Reference to store all fire markers (for later removal).
  const fireMarkersRef = useRef([]);
  // State to hold the search query.
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Define a style object that uses ESRI World Imagery for satellite imagery.
    const esriStyle = {
      version: 8,
      sources: {
        "esri-world-imagery": {
          type: "raster",
          tiles: [
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          ],
          tileSize: 256,
          attribution: "Tiles &copy; Esri"
        }
      },
      layers: [
        {
          id: "esri-world-imagery",
          type: "raster",
          source: "esri-world-imagery",
          minzoom: 0,
          maxzoom: 19
        }
      ]
    };

    // Initialize the map using ESRI imagery.
    mapRef.current = new maplibregl.Map({
      container: mapContainerRef.current,
      style: esriStyle,
      center: [-118.2437, 34.0522], // Center on Los Angeles.
      zoom: 8,
      pitch: 60,    // Add pitch for a 3D look.
      bearing: -10, // Slight rotation.
    });

    // Add navigation controls (zoom & rotation).
    mapRef.current.addControl(new maplibregl.NavigationControl(), 'top-right');

    // When the map loads, add a DEM source and enable terrain.
    mapRef.current.on('load', () => {
      // Add a raster-dem source using Terrarium tiles.
      mapRef.current.addSource('terrain-dem', {
        type: 'raster-dem',
        tiles: [
          "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
        ],
        tileSize: 256,
        maxzoom: 14
      });

      // Enable 3D terrain with a very low exaggeration factor to keep distortion minimal.
      mapRef.current.setTerrain({ source: 'terrain-dem', exaggeration: 0.05 });

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

    // Set up the click handler to add a fire marker and call your simulation endpoint.
    mapRef.current.on('click', async (event) => {
      const { lng, lat } = event.lngLat;
      console.log(`Map clicked at: ${lng}, ${lat}`);

      // Create a DOM element for the initial fire marker.
      const fireEl = document.createElement('div');
      fireEl.style.width = '12px';
      fireEl.style.height = '12px';
      fireEl.style.backgroundColor = 'orange';
      fireEl.style.borderRadius = '50%';
      fireEl.style.boxShadow = '0 0 8px 3px rgba(255, 140, 0, 0.8)';

      // Create the marker and add it to the map.
      const marker = new maplibregl.Marker({ element: fireEl, anchor: 'center' })
        .setLngLat([lng, lat])
        .addTo(mapRef.current);
      // Save the marker reference for later removal.
      fireMarkersRef.current.push(marker);

      // Send a POST request to your fire simulation endpoint.
      try {
        const response = await fetch('http://3.90.166.12:5000/fire_sim', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat, lon: lng }),
        });

        if (!response.ok) {
          console.error('Fire simulation API request failed:', response.statusText);
          return;
        }
        const points = await response.json();
        console.log('Fire simulation points:', points);

        // Plot each point with a short delay between each one.
        if (Array.isArray(points)) {
          // Set the delay (in milliseconds) between plotting each marker.
          const delay = 500; // 500ms delay

          for (let i = 0; i < points.length; i++) {
            const { lat: simLat, lon: simLon } = points[i];

            // Create a DOM element for the simulated fire marker.
            const simFireEl = document.createElement('div');
            simFireEl.style.width = '12px';
            simFireEl.style.height = '12px';
            simFireEl.style.backgroundColor = 'red'; // Use a different color for simulation markers.
            simFireEl.style.borderRadius = '50%';
            simFireEl.style.boxShadow = '0 0 8px 3px rgba(255, 69, 0, 0.8)';

            const simMarker = new maplibregl.Marker({ element: simFireEl, anchor: 'center' })
              .setLngLat([simLon, simLat])
              .addTo(mapRef.current);
            fireMarkersRef.current.push(simMarker);

            // Wait for the specified delay before plotting the next marker.
            await new Promise((resolve) => setTimeout(resolve, delay));
          }
        }
      } catch (error) {
        console.error('Error calling fire simulation endpoint:', error);
      }
    });

    // Cleanup the map instance on unmount.
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
      }
    };
  }, []);

  // Handler for submitting the search form using Nominatim for geocoding.
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(searchQuery)}&format=json`
      );
      if (!response.ok) {
        console.error('Geocoding request failed:', response.statusText);
        return;
      }
      const data = await response.json();
      console.log('Geocoding result:', data);

      if (data && data.length > 0) {
        const result = data[0];
        const lng = parseFloat(result.lon);
        const lat = parseFloat(result.lat);
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
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />

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
        <button onClick={handleResetFire} style={{ padding: '5px 10px', width: '100%' }}>
          Reset Fire
        </button>
      </div>
    </div>
  );
};

export default App;
