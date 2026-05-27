export function VestIcon({ className }: { className?: string }) {
  return (
    <span
      role="img"
      aria-label="Safety vest"
      className={className}
      style={{ fontSize: "1.35em", lineHeight: 1 }}
    >
      {"\uD83E\uDDBA"}
    </span>
  );
}
