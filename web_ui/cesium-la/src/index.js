// src/index.js
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { Ion } from 'cesium';

window.CESIUM_BASE_URL = process.env.REACT_APP_CESIUM_BASE_URL || '/cesium';

// Set your Cesium Ion access token (free registration required)
Ion.defaultAccessToken = process.env.REACT_APP_CESIUM_ION_TOKEN || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5ZTI5MmEyOS1hMWFhLTQ5NmMtYTQ5ZC0xMzM2NzY5ZGFmN2QiLCJpZCI6Mjc0MzQxLCJpYXQiOjE3Mzg5OTk4NzR9.Tt4_8tSE60sJwaXCwK2PwH4De1g3NiJI3lzNgCjPGU4';

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
