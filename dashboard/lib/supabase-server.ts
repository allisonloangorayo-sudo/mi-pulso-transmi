import { createClient } from "@supabase/supabase-js";

// Server-only: SUPABASE_KEY es la service_role/secret key. Nunca se importa
// desde un componente cliente ni se expone con el prefijo NEXT_PUBLIC_.
export function getSupabaseServer() {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_KEY;
  if (!url || !key) {
    throw new Error("SUPABASE_URL/SUPABASE_KEY no configurados en el entorno del servidor.");
  }
  return createClient(url, key, { auth: { persistSession: false } });
}
