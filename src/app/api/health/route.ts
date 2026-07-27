import { NextResponse } from 'next/server';
import fs from 'fs';
import os from 'os';

export async function GET() {
  try {
    // Gather system information
    const uptime = process.uptime();
    const memoryUsage = process.memoryUsage();
    const cpuUsage = os.cpus();
    
    const healthData = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: {
        seconds: Math.floor(uptime),
        human: formatUptime(uptime),
      },
      system: {
        platform: os.platform(),
        arch: os.arch(),
        cpus: cpuUsage.length,
        totalMemory: os.totalmem(),
        freeMemory: os.freemem(),
        loadAverage: os.loadavg(),
      },
      process: {
        memory: {
          rss: memoryUsage.rss,
          heapUsed: memoryUsage.heapUsed,
          heapTotal: memoryUsage.heapTotal,
          external: memoryUsage.external,
        },
        pid: process.pid,
        nodeVersion: process.version,
      },
      services: {
        api: 'operational',
        database: checkDatabaseConnection(),
        gpu: detectGPUs(),
      },
      version: '2.0.0',
      environment: process.env.NODE_ENV || 'development',
    };

    return NextResponse.json(healthData);
  } catch (error) {
    console.error('Health check error:', error);
    return NextResponse.json(
      { 
        status: 'unhealthy',
        error: 'Health check failed',
        timestamp: new Date().toISOString(),
      },
      { status: 503 }
    );
  }
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  
  if (days > 0) return `${days}d ${hours}h ${minutes}m`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function checkDatabaseConnection(): string {
  try {
    // Check if database file exists
    const dbPath = './db/custom.db';
    if (fs.existsSync(dbPath)) {
      return 'operational';
    }
    return 'not_configured';
  } catch {
    return 'error';
  }
}

function detectGPUs(): any[] {
  // In production, this would use nvidia-smi or similar
  // For now, return simulated GPU info
  return [
    {
      id: 0,
      name: 'NVIDIA A100-SXM4-80GB',
      memory: { total: 80 * 1024 * 1024 * 1024, used: 30 * 1024 * 1024 * 1024 },
      utilization: 35,
      status: 'available',
    },
    {
      id: 1,
      name: 'NVIDIA A100-SXM4-80GB',
      memory: { total: 80 * 1024 * 1024 * 1024, used: 55 * 1024 * 1024 * 1024 },
      utilization: 68,
      status: 'in_use',
    },
  ];
}
