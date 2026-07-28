import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    success: true,
    data: {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      service: 'GLM Platform',
      features: [
        'chat',
        'image-generation',
        'vision',
        'web-search',
        'tts',
        'asr'
      ],
      backend: process.env.NEXT_PUBLIC_API_URL || 'https://conversationalglm2.onrender.com'
    },
  });
}

export async function HEAD() {
  return new Response(null, { status: 200 });
}
