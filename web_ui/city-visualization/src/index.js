// src/index.js
import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

// Find the root container where your app will be rendered.
const container = document.getElementById('root');

if (!container) {
  throw new Error("Failed to find the root element. Make sure there is an element with id='root' in your index.html");
}

// Create a root.
const root = createRoot(container);

// Render your App.
root.render(<App />);
