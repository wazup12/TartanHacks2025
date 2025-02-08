// src/App.js
import React, { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const App = () => {
  // Reference for the map container element.
  const mapContainerRef = useRef(null);
  // Reference to store the MapLibre map instance.
  const mapRef = useRef(null);
  // Reference to store all markers/overlays (for later removal).
  const markersRef = useRef([]);
  // Reference to store the overlay GeoJSON data (for the static PNG overlay).
  const overlayDataRef = useRef({
    type: 'FeatureCollection',
    features: []
  });
  // State to hold the search query.
  const [searchQuery, setSearchQuery] = useState('');

  // -------------------------
  // Wind field function
  // -------------------------
  const getWindAtCoord = (lat, lon) => {
    const centerLat = 34.0522;
    const centerLon = -118.2437;
    const magnitude = 0.5 + 0.3 * Math.sin((lat - centerLat) * Math.PI / 5);
    const angle = 45 + 20 * Math.cos((lon - centerLon) * Math.PI / 5);
    return { magnitude, angle };
  };

  // -------------------------
  // Create wind marker element (for visualization)
  // -------------------------
  const createWindMarkerElement = (wind) => {
    const el = document.createElement('div');
    el.style.width = '20px';
    el.style.height = '20px';
    el.style.fontSize = '16px';
    el.style.color = 'blue';
    el.style.textAlign = 'center';
    el.style.lineHeight = '20px';
    el.innerText = '➤';
    el.style.transform = `rotate(${wind.angle}deg)`;
    return el;
  };

  const addOverlay = (lng, lat, overlayImageUrl) => {
    if (mapRef.current.getLayer('overlay-layer')) {
      mapRef.current.removeLayer('overlay-layer');
    }
    if (mapRef.current.getSource('overlay-source')) {
      mapRef.current.removeSource('overlay-source');
    }
    const deltaWidth = 0.023;
    const deltaHeight = 0.023;
    const topLeft = [lng - deltaWidth / 2, lat + deltaHeight / 2];
    const topRight = [lng + deltaWidth / 2, lat + deltaHeight / 2];
    const bottomRight = [lng + deltaWidth / 2, lat - deltaHeight / 2];
    const bottomLeft = [lng - deltaWidth / 2, lat - deltaHeight / 2];
    const coordinates = [topLeft, topRight, bottomRight, bottomLeft];

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = overlayImageUrl;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] === 0 && data[i + 1] === 0 && data[i + 2] === 0) {
          data[i + 3] = 0;
        }
      }
      ctx.putImageData(imageData, 0, 0);
      const dataUrl = canvas.toDataURL();

      mapRef.current.addSource('overlay-source', {
        type: 'image',
        url: dataUrl,
        coordinates: coordinates
      });
      mapRef.current.addLayer({
        id: 'overlay-layer',
        type: 'raster',
        source: 'overlay-source',
        paint: { 'raster-opacity': 1 }
      });
    };
    img.onerror = (error) => {
      console.error('Error loading overlay image:', error);
    };
  };

  const addAnimatedGifOverlay = (lng, lat, gifUrl) => {
    const imgEl = document.createElement('img');
    imgEl.src = gifUrl;
    imgEl.alt = 'Animated Overlay';
    imgEl.style.width = '900px';
    imgEl.style.height = '900px';
    const gifMarker = new maplibregl.Marker({ element: imgEl, anchor: 'center' })
      .setLngLat([lng, lat])
      .addTo(mapRef.current);
    markersRef.current.push(gifMarker);
  };

  // -------------------------
  // Map Initialization and Event Handlers
  // -------------------------
  useEffect(() => {
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

    mapRef.current = new maplibregl.Map({
      container: mapContainerRef.current,
      style: esriStyle,
      center: [-118.2437, 34.0522],
      zoom: 8,
      pitch: 60,
      bearing: -10,
    });

    mapRef.current.addControl(new maplibregl.NavigationControl(), 'top-right');

    mapRef.current.on('load', () => {
      overlayDataRef.current = {
        type: 'FeatureCollection',
        features: []
      };

      // Add a grid of wind gradient markers.
      const bounds = mapRef.current.getBounds();
      const north = bounds.getNorth();
      const south = bounds.getSouth();
      const east = bounds.getEast();
      const west = bounds.getWest();
      const numRows = 5;
      const numCols = 5;
      const latStep = (north - south) / (numRows - 1);
      const lonStep = (east - west) / (numCols - 1);
      for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numCols; j++) {
          const latPos = south + i * latStep;
          const lonPos = west + j * lonStep;
          const wind = getWindAtCoord(latPos, lonPos);
          const windEl = createWindMarkerElement(wind);
          const windMarker = new maplibregl.Marker({ element: windEl, anchor: 'center' })
            .setLngLat([lonPos, latPos])
            .addTo(mapRef.current);
          markersRef.current.push(windMarker);
        }
      }
    });

    mapRef.current.on('click', async (event) => {
      const { lng, lat } = event.lngLat;
      console.log(`Map clicked at: ${lng}, ${lat}`);

      const windAtClick = getWindAtCoord(lat, lng);
      console.log(`Wind at click: magnitude = ${windAtClick.magnitude.toFixed(2)}, angle = ${windAtClick.angle.toFixed(2)}`);
      try {
        const streetResponse = await fetch('http://3.90.166.12:5000/street_images', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat, lon: lng })
        });
        if (!streetResponse.ok) {
          console.error('Street images endpoint error:', streetResponse.statusText);
          return;
        }
        const streetData = await streetResponse.json();
        console.log('Street images data:', streetData);
        const filename = streetData.overlay_image;
        const overlayImageUrl = "http://3.90.166.12:5000/" + filename;
        console.log('Static overlay image URL:', overlayImageUrl);
        addOverlay(lng, lat, overlayImageUrl);
      } catch (error) {
        console.error('Error calling street_images endpoint:', error);
      }

      try {
        const gifResponse = await fetch('http://3.90.166.12:5000/gen_gif', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat, lon: lng })
        });
        if (!gifResponse.ok) {
          console.error('gen_gif endpoint error:', gifResponse.statusText);
          return;
        }
        const gifData = await gifResponse.json();
        console.log('gen_gif data:', gifData);
        const gifFilename = gifData.gif_path;
        const animatedGifUrl = "http://3.90.166.12:5000/" + gifFilename;
        console.log('Animated GIF overlay URL:', animatedGifUrl);
        addAnimatedGifOverlay(lng, lat, animatedGifUrl);
      } catch (error) {
        console.error('Error calling gen_gif endpoint:', error);
      }

      // --- Add the initial fire marker using a fire emoji (made transparent) ---
      const initEl = document.createElement('div');
      initEl.innerText = '🔥';
      initEl.style.fontSize = '24px';
      initEl.style.opacity = '0.5';  // Make the fire emoji transparent
      const initMarker = new maplibregl.Marker({ element: initEl, anchor: 'center' })
        .setLngLat([lng, lat])
        .addTo(mapRef.current);
      markersRef.current.push(initMarker);

      // --- Call the simulation endpoint with wind data ---
      try {
        const response = await fetch('http://3.90.166.12:5000/fire_sim', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            lat,
            lon: lng,
            mag: windAtClick.magnitude,
            ang: windAtClick.angle
          }),
        });
        if (!response.ok) {
          console.error('Simulation endpoint error:', response.statusText);
          return;
        }
        const simData = await response.json();
        console.log('Simulation data:', simData);

        const timeSeries = simData.time_series_positions;
        for (let t = 0; t < timeSeries.length; t++) {
          let positions = timeSeries[t].filter((_, idx) => idx % 10 === 0);
          positions.forEach((pos) => {
            const [simLat, simLon] = pos;
            const markerEl = document.createElement('div');
            markerEl.innerText = '🔥';
            markerEl.style.fontSize = '24px';
            markerEl.style.opacity = '0.5'; // Make simulation fire emojis transparent
            const marker = new maplibregl.Marker({ element: markerEl, anchor: 'center' })
              .setLngLat([simLon, simLat])
              .addTo(mapRef.current);
            markersRef.current.push(marker);
          });
          await new Promise((resolve) => setTimeout(resolve, 100));
        }
      } catch (error) {
        console.error('Error calling simulation endpoint:', error);
      }
    });

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
      }
    };
  }, []);

  // -------------------------
  // Search and Reset Handlers
  // -------------------------
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    try {
      const response = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(searchQuery)}&format=json`);
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

  const handleReset = () => {
    if (mapRef.current.getLayer('overlay-layer')) {
      mapRef.current.removeLayer('overlay-layer');
    }
    if (mapRef.current.getSource('overlay-source')) {
      mapRef.current.removeSource('overlay-source');
    }
    if (mapRef.current.getLayer('animated-overlay-layer')) {
      mapRef.current.removeLayer('animated-overlay-layer');
    }
    if (mapRef.current.getSource('animated-overlay-source')) {
      mapRef.current.removeSource('animated-overlay-source');
    }
    overlayDataRef.current = {
      type: 'FeatureCollection',
      features: []
    };
    markersRef.current.forEach((marker) => marker.remove());
    markersRef.current = [];
  };

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />
      <div
        style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          backgroundColor: 'white',
          padding: '10px',
          borderRadius: '4px',
          boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
          zIndex: 1,
        }}
      >
        <form onSubmit={handleSearch} style={{ display: 'flex', marginBottom: '10px' }}>
          <input
            type="text"
            placeholder="Search for a location..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ flex: 1, marginRight: '5px', padding: '5px' }}
          />
          <button type="submit" style={{ padding: '5px 10px' }}>
            Search
          </button>
        </form>
        <button onClick={handleReset} style={{ padding: '5px 10px', width: '100%' }}>
          Reset Fire
        </button>
      </div>
    </div>
  );
};

export default App;
