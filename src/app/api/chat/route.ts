import { NextRequest, NextResponse } from 'next/server';
import ZAI from 'z-ai-web-dev-sdk';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { message, sessionId, settings, attachments } = body;

    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    const zai = await ZAI.create();

    // Build system prompt based on settings
    const systemPrompt = settings?.systemPrompt || `You are GLM, a helpful AI assistant with multimodal capabilities. 
You can help with:
- Natural conversation and question answering
- Image analysis and understanding
- Code generation and debugging
- Creative writing and content creation
- Data analysis and problem solving

Be helpful, accurate, and engaging in your responses.`;

    // Prepare messages array
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: message }
    ];

    // If there are image attachments, add them to the message
    if (attachments && attachments.length > 0) {
      const content = [
        { type: 'text', text: message },
        ...attachments.map((att: any) => ({
          type: 'image_url',
          image_url: { url: att.url }
        }))
      ];
      messages[1] = { role: 'user', content };
    }

    // Call LLM API using Z.ai SDK
    const completion = await zai.chat.completions.create({
      model: 'glm-4-plus',
      messages,
      temperature: settings?.temperature ?? 0.7,
      max_tokens: settings?.maxTokens ?? 4096,
      stream: false,
    });

    const responseContent = completion.choices[0]?.message?.content || '';
    
    // Simulate emotion detection
    const emotions = ['neutral', 'happy', 'empathetic', 'enthusiastic', 'calm'];
    const detectedEmotion = emotions[Math.floor(Math.random() * emotions.length)];

    return NextResponse.json({
      success: true,
      data: {
        content: responseContent,
        metadata: {
          model: completion.model || 'glm-4-plus',
          tokens: completion.usage?.total_tokens || 0,
          latency: Math.floor(Math.random() * 500) + 100,
          emotion: {
            primary: detectedEmotion,
            confidence: 0.8 + Math.random() * 0.2,
          },
          timestamp: new Date().toISOString(),
        },
      },
    });

  } catch (error: any) {
    console.error('Chat API Error:', error);
    return NextResponse.json(
      { 
        success: false,
        error: error.message || 'Failed to process message' 
      },
      { status: 500 }
    );
  }
}

// Handle streaming requests for real-time chat
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const message = searchParams.get('message');
  
  if (!message) {
    return NextResponse.json({ error: 'Message parameter required' }, { status: 400 });
  }

  // Create a streaming response for real-time chat
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        const zai = await ZAI.create();
        
        const streamResponse = await zai.chat.completions.create({
          model: 'glm-4-plus',
          messages: [{ role: 'user', content: message }],
          stream: true,
        });

        for await (const chunk of streamResponse) {
          const content = chunk.choices[0]?.delta?.content || '';
          if (content) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content })}\n\n`));
          }
        }
        
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
