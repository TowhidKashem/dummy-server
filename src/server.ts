import { z } from "zod";
import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { zValidator } from "@hono/zod-validator";
import { smoothStream, streamText } from "ai";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";

const app = new Hono<{
  Bindings: {
    OPEN_ROUTER_API_KEY: string;
  };
}>();

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
            role: z.enum(["user", "assistant", "system"]),
            content: z.string().min(1, "Message content cannot be empty"),
          })
        )
        .min(1, "Messages array cannot be empty"),
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
        model: openrouter("google/gemini-2.0-flash-001"),
        messages: messages,
        experimental_transform: smoothStream({
          delayInMs: 50,
          chunking: "word",
        }),
      });

      return result.toTextStreamResponse({
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-cache",
          "content-encoding": "identity",
          "transfer-encoding": "chunked",
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

app.get("/ping", (ctx) => {
  console.log("pong");
  return ctx.text("pong");
});

app.all("*", (ctx) => {
  console.log(`404 - Route not found: ${ctx.req.method} ${ctx.req.path}`);
  throw new HTTPException(404, {
    message: `Route ${ctx.req.method} ${ctx.req.path} not found`,
  });
});

export default app;
