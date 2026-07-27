import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Allowed file types and their max sizes
const ALLOWED_TYPES: Record<string, { maxSize: number; extensions: string[] }> = {
  image: { 
    maxSize: 10 * 1024 * 1024, // 10MB
    extensions: ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
  },
  audio: { 
    maxSize: 50 * 1024 * 1024, // 50MB
    extensions: ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
  },
  video: { 
    maxSize: 500 * 1024 * 1024, // 500MB
    extensions: ['.mp4', '.webm', '.mov', '.avi', '.mkv']
  },
  document: { 
    maxSize: 25 * 1024 * 1024, // 25MB
    extensions: ['.pdf', '.doc', '.docx', '.txt', '.md', '.json', '.csv']
  },
  dataset: { 
    maxSize: 5 * 1024 * 1024 * 1024, // 5GB for datasets
    extensions: ['.jsonl', '.json', '.csv', '.parquet', '.zip', '.tar.gz']
  },
};

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const type = (formData.get('type') as string) || 'image';
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Validate file type
    const config = ALLOWED_TYPES[type];
    if (!config) {
      return NextResponse.json(
        { error: `Invalid file type. Allowed types: ${Object.keys(ALLOWED_TYPES).join(', ')}` },
        { status: 400 }
      );
    }

    // Get file extension
    const extension = path.extname(file.name).toLowerCase();
    
    // Validate extension
    if (!config.extensions.includes(extension)) {
      return NextResponse.json(
        { error: `Invalid file extension. Allowed: ${config.extensions.join(', ')}` },
        { status: 400 }
      );
    }

    // Validate file size
    if (file.size > config.maxSize) {
      return NextResponse.json(
        { error: `File too large. Maximum size: ${formatBytes(config.maxSize)}` },
        { status: 400 }
      );
    }

    // Create upload directory
    const uploadDir = path.join(process.cwd(), 'public', 'uploads', type);
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    // Generate unique filename
    const uniqueName = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}${extension}`;
    const filePath = path.join(uploadDir, uniqueName);

    // Save file
    const buffer = Buffer.from(await file.arrayBuffer());
    fs.writeFileSync(filePath, buffer);

    return NextResponse.json({
      success: true,
      file: {
        name: file.name,
        originalName: file.name,
        filename: uniqueName,
        url: `/uploads/${type}/${uniqueName}`,
        size: file.size,
        type: file.type,
        extension,
        uploadedAt: new Date().toISOString(),
      },
    });

  } catch (error: any) {
    console.error('Upload Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to upload file' },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type');
    
    const baseDir = path.join(process.cwd(), 'public', 'uploads');
    let files: any[] = [];
    
    if (fs.existsSync(baseDir)) {
      const typesToScan = type ? [type] : Object.keys(ALLOWED_TYPES);
      
      for (const scanType of typesToScan) {
        const typeDir = path.join(baseDir, scanType);
        if (fs.existsSync(typeDir)) {
          const typeFiles = fs.readdirSync(typeDir)
            .filter(f => !f.startsWith('.'))
            .map(f => ({
              filename: f,
              url: `/uploads/${scanType}/${f}`,
              type: scanType,
              size: fs.statSync(path.join(typeDir, f)).size,
            }));
          files.push(...typeFiles);
        }
      }
    }

    return NextResponse.json({
      success: true,
      files: files.sort((a, b) => b.filename.localeCompare(a.filename)),
    });

  } catch (error: any) {
    console.error('List Uploads Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to list uploads' },
      { status: 500 }
    );
  }
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
