import { Api, getToken, setToken } from "./api";

export type Session = {
  authenticated: boolean;
  user: any | null;
  workspace?: any | null;
};

export const isLoggedIn = () => !!getToken();

export async function register(body: { email: string; password: string; full_name?: string; organisation?: string }) {
  const res = await Api.register(body);
  setToken(res.token);
  return res;
}

export async function login(email: string, password: string) {
  const res = await Api.login({ email, password });
  setToken(res.token);
  return res;
}

export function logout() {
  Api.logout().catch(() => undefined);
  setToken(null);
}

export async function whoami(): Promise<Session> {
  try { return await Api.me(); } catch { return { authenticated: false, user: null }; }
}
