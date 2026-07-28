import { NextRequest, NextResponse } from 'next/server';

// Middleware runs on all requests (both Netlify and Render)
// This ensures CORS headers are present at the edge level
export function middleware(request: NextRequest) {
  const response = NextResponse.next();

  // Get the origin from request
  const origin = request.headers.get('origin') || '';

  // Allow specific origins or all origins for development
  const allowedOrigins = [
    'https://conversationalglm.netlify.app',  // Your Netlify frontend
    'http://localhost:3000',                   // Local development
    'https://conversationalglm2.onrender.com', // Your Render backend
  ];

  // Check if origin is allowed or if it's a same-origin request
  const isAllowedOrigin = allowedOrigins.some(o => origin === o) || 
                          !origin || 
                          origin.includes('localhost');

  if (isAllowedOrigin) {
    response.headers.set('Access-Control-Allow-Origin', origin || '*');
  }

  // Add CORS headers for API routes
  if (request.nextUrl.pathname.startsWith('/api/')) {
    response.headers.set('Access-Control-Allow-Origin', '*');
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');

    // Handle preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
        },
      });
    }
  }

  return response;
}

// Configure which paths the middleware should run on
export const config = {
  matcher: [
    '/api/:path*',
    '/:path*',  // Apply to all paths for security headers
  ],
};
