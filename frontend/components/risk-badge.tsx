import { Badge } from "@/components/ui/badge";

const RISK_STYLES: Record<string, string> = {
  CRITICAL: "bg-red-600 text-white hover:bg-red-600",
  HIGH: "bg-orange-500 text-white hover:bg-orange-500",
  MEDIUM: "bg-yellow-500 text-black hover:bg-yellow-500",
  LOW: "bg-blue-500 text-white hover:bg-blue-500",
};

export function RiskBadge({ level }: { level: string }) {
  const cls = RISK_STYLES[(level || "").toUpperCase()] ?? "bg-gray-500 text-white hover:bg-gray-500";
  return <Badge className={cls}>{level}</Badge>;
}
