import { z } from "zod";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { secureHeaders } from "hono/secure-headers";
import { HTTPException } from "hono/http-exception";
import { zValidator } from "@hono/zod-validator";
import { smoothStream, streamText } from "ai";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const app = new Hono<{
  Bindings: {
    OPEN_ROUTER_API_KEY: string;
  };
}>();

// Apply CORS middleware - allow all origins for public API
app.use(
  "*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization", "X-Requested-With"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    credentials: true,
    maxAge: 86400,
  })
);

// Apply secure headers middleware
app.use("*", secureHeaders());

// Global error handler
app.onError((err, ctx) => {
  console.error("Global error handler:", err);

  if (err instanceof HTTPException) {
    return ctx.json(
      {
        error: "HTTP Exception",
        message: err.message,
      },
      err.status
    );
  }

  return ctx.json(
    {
      error: "Internal Server Error",
      message:
        err instanceof Error ? err.message : "An unexpected error occurred",
    },
    500
  );
});

app.post(
  "/chat",
  zValidator(
    "json",
    z.object({
      messages: z
        .array(
          z.object({
            role: z.enum(["user", "assistant"]),
            content: z.string().min(1, "Message content cannot be empty"),
          })
        )
        .min(1, "Messages array cannot be empty")
        .refine(validateMessagePattern, {
          message:
            "Messages must alternate between user and assistant roles, and the last message must be from the user.",
        }),
    }),
    (result, c) => {
      if (!result.success) {
        return c.json(
          {
            error: "Validation Error",
            details: result.error.issues.map((issue) => ({
              path: issue.path.join("."),
              message: issue.message,
              code: issue.code,
            })),
          },
          400
        );
      }
    }
  ),
  async (ctx) => {
    try {
      const { messages } = ctx.req.valid("json");

      console.log("chat endpoint hit");
      console.log("Received messages:", messages);

      const openrouter = createOpenRouter({
        apiKey: ctx.env.OPEN_ROUTER_API_KEY,
      });

      const result = streamText({
        model: openrouter("deepseek/deepseek-chat-v3.1:free"),
        messages: [
          {
            role: "system",
            content: `You are a helpful assistant. Keep all your responses on topic and professional. Do not deviate from the question being asked. Return your responses in markdown format.`,
          },
          ...messages,
        ],
        experimental_transform: smoothStream({
          delayInMs: 50,
          chunking: "word",
        }),
      });

      // Create a custom SSE stream from the AI SDK result
      const encoder = new TextEncoder();

      const stream = new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of result.textStream) {
              // Send each chunk in SSE format
              const sseData = `data: ${JSON.stringify({ content: chunk })}\n\n`;
              controller.enqueue(encoder.encode(sseData));
            }

            // Send the [DONE] signal
            const doneSignal = `data: [DONE]\n\n`;
            controller.enqueue(encoder.encode(doneSignal));

            controller.close();
          } catch (error) {
            console.error("Stream error:", error);
            const errorData = `data: ${JSON.stringify({
              error: "Stream error occurred",
            })}\n\n`;
            controller.enqueue(encoder.encode(errorData));
            controller.close();
          }
        },
      });

      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    } catch (error) {
      console.error(error);
      return ctx.json(
        {
          error: "Internal Server Error",
          message:
            error instanceof Error
              ? error.message
              : "An unexpected error occurred",
        },
        500
      );
    }
  }
);

app.get("/health", (ctx) => {
  return ctx.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
  });
});

// Handle OPTIONS requests for CORS preflight
app.options("*", (ctx) => ctx.text(""));

app.all("*", (ctx) => {
  console.log(`404 - Route not found: ${ctx.req.method} ${ctx.req.path}`);
  throw new HTTPException(404, {
    message: `Route ${ctx.req.method} ${ctx.req.path} not found`,
  });
});

export default app;

function validateMessagePattern(messages: Message[]): boolean {
  // Check if the last message is from the user
  const lastMessage = messages.at(-1);
  if (lastMessage?.role !== "user") {
    return false;
  }

  // Check if messages alternate between user and assistant
  for (let i = 1; i < messages.length; i++) {
    if (messages[i].role === messages[i - 1].role) {
      return false;
    }
  }

  // Verify the pattern is correct based on message count
  // Odd number of messages: should start with "user"
  // Even number of messages: should start with "assistant"
  const isOddCount = messages.length % 2 === 1;
  const firstMessage = messages[0];

  if (isOddCount && firstMessage.role !== "user") {
    return false;
  }
  if (!isOddCount && firstMessage.role !== "assistant") {
    return false;
  }

  return true;
}
