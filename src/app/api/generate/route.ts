import { NextRequest, NextResponse } from 'next/server';
import ZAI from 'z-ai-web-dev-sdk';
import fs from 'fs';
import path from 'path';

const SUPPORTED_SIZES = [
  '1024x1024',
  '768x1344',
  '864x1152',
  '1344x768',
  '1152x864',
  '1440x720',
  '720x1440',
];

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { prompt, size = '1024x1024', count = 1 } = body;

    if (!prompt) {
      return NextResponse.json(
        { error: 'Prompt is required' },
        { status: 400 }
      );
    }

    if (!SUPPORTED_SIZES.includes(size)) {
      return NextResponse.json(
        { error: `Invalid size. Supported sizes: ${SUPPORTED_SIZES.join(', ')}` },
        { status: 400 }
      );
    }

    const zai = await ZAI.create();

    // Ensure output directory exists
    const outputDir = path.join(process.cwd(), 'public', 'images', 'generated');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const results = [];

    for (let i = 0; i < Math.min(count, 4); i++) {
      const response = await zai.images.generations.create({
        prompt,
        size,
      });

      if (!response.data || !response.data[0] || !response.data[0].base64) {
        throw new Error('Invalid response from image generation API');
      }

      const imageBase64 = response.data[0].base64;
      const buffer = Buffer.from(imageBase64, 'base64');
      
      // Generate unique filename
      const filename = `img_${Date.now()}_${i}.png`;
      const filepath = path.join(outputDir, filename);
      
      fs.writeFileSync(filepath, buffer);

      results.push({
        success: true,
        url: `/images/generated/${filename}`,
        filename,
        size: buffer.length,
        dimensions: size,
      });
    }

    return NextResponse.json({
      success: true,
      images: results,
      prompt,
    });

  } catch (error: any) {
    console.error('Image Generation API Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to generate image' },
      { status: 500 }
    );
  }
}

// Get generated images list
export async function GET() {
  try {
    const outputDir = path.join(process.cwd(), 'public', 'images', 'generated');
    
    let files: string[] = [];
    if (fs.existsSync(outputDir)) {
      files = fs.readdirSync(outputDir)
        .filter(f => f.endsWith('.png') || f.endsWith('.jpg'))
        .map(f => `/images/generated/${f}`);
    }

    return NextResponse.json({ 
      success: true,
      images: files.slice(-50), // Return last 50 images
    });
  } catch (error) {
    console.error('Error listing images:', error);
    return NextResponse.json(
      { error: 'Failed to list images' },
      { status: 500 }
    );
  }
}
